import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from pytorch_metric_learning import distances, losses

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin, StableDiffusionXLLoraLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

logger = logging.get_logger(__name__)


class GaussianSmoothing(torch.nn.Module):
    """
    Apply Gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel using
    a depthwise convolution.
    """
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # Build Gaussian kernel
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class AttentionStore:
    """
    Stores cross-attention maps for each UNet layer at each timestep.
    """
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __init__(self, attn_res: Tuple[int, int]):
        self.attn_res = attn_res
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __call__(self, attn: Attention, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self) -> Dict[str, List[torch.Tensor]]:
        return self.attention_store

    def aggregate_attention(self, from_where: Tuple[str, ...]) -> torch.Tensor:
        """
        Aggregates attention across heads and layers at specified resolutions.
        """
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps.get(location, []):
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionProcessor:
    """
    Wraps the original Attention module to store cross-attention maps into AttentionStore.
    """
    def __init__(self, attnstore: AttentionStore, place_in_unet: str):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask_prepared = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask_prepared)

        # Store only cross-attention
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def rescale_noise_cfg(noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float = 0.0) -> torch.Tensor:
    """
    Rescales `noise_cfg` based on `guidance_rescale` to correct overexposure (SDXL-specific).
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler: KarrasDiffusionSchedulers,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """
    Retrieves timesteps from scheduler (copied from SDXL pipeline).
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom `timesteps`."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom `sigmas`."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ConformXLPipeline(StableDiffusionXLPipeline, TextualInversionLoaderMixin):
    """
    Conform (Attend-and-Contrast) adapted for Stable Diffusion XL.

    Implements contrastive attention-based refinement to enforce better object separation in SDXL.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            image_encoder=image_encoder,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        # ------------------------- 메모리 절약을 위한 옵션들 -------------------------
        # 1) UNet attention slicing 활성화 → 한 번에 들어오는 attention 맵 크기를 줄여 VRAM 피크 감소
        # self.unet.enable_attention_slicing()

        # 2) Gradient checkpointing 활성화 → 중간 activation을 저장하지 않고 필요 시 재연산
        self.unet.enable_gradient_checkpointing()

        # 3) (이미 .from_pretrained(torch_dtype=torch.float16) 로 로드한다고 가정)
        #    UNet, VAE, CLIP 모델이 이미 fp16. 만일 아니라면 강제로 fp16으로 변환:
        # self.unet.to(torch.float16)
        # self.vae.to(torch.float16)
        # self.text_encoder.to(torch.float16)
        # self.text_encoder_2.to(torch.float16)

        # 4) CPU 오프로딩: 큰 모델들을 VRAM 대신 CPU로 옮겨두었다가
        #    inference 직전에 GPU로 로드 → peak VRAM 사용량 절감
        # self.enable_sequential_cpu_offload()
        # -------------------------------------------------------------------------

    @property
    def components(self) -> Dict[str, torch.nn.Module]:
        """
        Only return the exact seven modules that StableDiffusionXLPipeline expects:
        'scheduler', 'vae', 'unet', 'text_encoder', 'tokenizer', 'tokenizer_2', 'text_encoder_2'.
        """
        return {
            "scheduler": self.scheduler,
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "text_encoder_2": self.text_encoder_2,
        }

    def register_attention_control(self) -> None:
        """
        Replace UNet's cross-attention processors with our AttentionProcessor so we can capture cross-attention maps.
        """
        attn_procs = {}
        cross_att_count = 0
        for name, proc in self.unet.attn_processors.items():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttentionProcessor(
                attnstore=self.attention_store,
                place_in_unet=place_in_unet,
            )
        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @staticmethod
    def _compute_contrastive_loss(
        attention_maps: torch.Tensor,
        attention_maps_t_plus_one: Optional[torch.Tensor],
        token_groups: List[List[int]],
        loss_type: str,
        temperature: float = 0.07,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
    ) -> torch.Tensor:
        """
        Computes the contrastive loss over cross-attention maps for grouped tokens.
        """

        # Remove start/end tokens: [batch, heads, H*W, seq_len] -> take [:, :, 1:-1]
        attention_for_text = attention_maps[:, :, 1:-1]

        if softmax_normalize:
            attention_for_text = attention_for_text * 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        attention_for_text_t_plus_one = None
        if attention_maps_t_plus_one is not None:
            attention_for_text_t_plus_one = attention_maps_t_plus_one[:, :, 1:-1]
            if softmax_normalize:
                attention_for_text_t_plus_one = attention_for_text_t_plus_one * 100
                attention_for_text_t_plus_one = torch.nn.functional.softmax(
                    attention_for_text_t_plus_one, dim=-1
                )

        # Map each token index to a class ID
        indices_to_classes: Dict[int, int] = {}
        for c, group in enumerate(token_groups):
            for idx in group:
                indices_to_classes[idx] = c

        classes_list = []
        embeddings_list = []
        for idx, c in indices_to_classes.items():
            classes_list.append(c)
            # Shift index because we've removed the first token
            embedding = attention_for_text[:, :, idx - 1]
            if do_smoothing:
                smoothing = GaussianSmoothing(
                    channels=1,
                    kernel_size=smoothing_kernel_size,
                    sigma=smoothing_sigma,
                    dim=2,
                ).to(attention_for_text.device)
                inp = F.pad(embedding.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                embedding = smoothing(inp).squeeze(0).squeeze(0)
            embedding = embedding.view(-1)

            if softmax_normalize_attention_maps:
                embedding = embedding * 100
                embedding = torch.nn.functional.softmax(embedding, dim=-1)
            embeddings_list.append(embedding)

            if attention_for_text_t_plus_one is not None:
                classes_list.append(c)
                emb_next = attention_for_text_t_plus_one[:, :, idx - 1]
                if do_smoothing:
                    smoothing = GaussianSmoothing(
                        channels=1,
                        kernel_size=smoothing_kernel_size,
                        sigma=smoothing_sigma,
                        dim=2,
                    ).to(attention_for_text.device)
                    inp = F.pad(emb_next.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                    emb_next = smoothing(inp).squeeze(0).squeeze(0)
                emb_next = emb_next.view(-1)
                if softmax_normalize_attention_maps:
                    emb_next = emb_next * 100
                    emb_next = torch.nn.functional.softmax(emb_next, dim=-1)
                embeddings_list.append(emb_next)

        classes_tensor = torch.tensor(classes_list).to(attention_for_text.device)
        embeddings_tensor = torch.stack(embeddings_list, dim=0).to(attention_for_text.device)

        if loss_type == "ntxent_contrastive":
            if len(token_groups) > 0 and len(token_groups[0]) > 1:
                loss_fn = losses.NTXentLoss(temperature=temperature)
            else:
                loss_fn = losses.ContrastiveLoss(
                    distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
                )
        elif loss_type == "ntxent":
            loss_fn = losses.NTXentLoss(temperature=temperature)
        elif loss_type == "contrastive":
            loss_fn = losses.ContrastiveLoss(
                distance=distances.CosineSimilarity(), pos_margin=1, neg_margin=0
            )
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        loss = loss_fn(embeddings_tensor, classes_tensor)
        return loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor,
        loss: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        """
        Updates latents in-place using gradient of the contrastive loss, then detaches to clear graph.
        """
        grad_cond = torch.autograd.grad(loss, latents, create_graph=False)[0]
        grad_cond = torch.nan_to_num(grad_cond, nan=0.0, posinf=1e3, neginf=-1e3) # nan/inf 방지
        latents.data.sub_(step_size * grad_cond)
        return latents.detach().requires_grad_(True)

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        token_groups: List[List[int]],
        loss: torch.Tensor,
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        refinement_steps: int = 20,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        temperature: float = 0.07,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        attention_maps_t_plus_one: Optional[torch.Tensor] = None,
        loss_fn: str = "ntxent",
        added_cond_kwargs = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively refines latents to minimize contrastive loss until threshold or steps exhausted.
        """
        for iteration_i in range(refinement_steps):
            latents = latents.detach().requires_grad_(True)
            # Forward with gradient tracking
            self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=self.ca_kwargs,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            self.unet.zero_grad()

            attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"))
            loss = self._compute_contrastive_loss(
                attention_maps=attn_maps,
                attention_maps_t_plus_one=attention_maps_t_plus_one,
                token_groups=token_groups,
                loss_type=loss_fn,
                do_smoothing=do_smoothing,
                temperature=temperature,
                smoothing_kernel_size=smoothing_kernel_size,
                smoothing_sigma=smoothing_sigma,
                softmax_normalize=softmax_normalize,
                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
            )
            del attn_maps
            torch.cuda.empty_cache()  # ← 매 iteration마다 불필요한 캐시 비우기

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)
            del loss
            torch.cuda.empty_cache()

        # Final forward to compute final loss
        latents = latents.detach().requires_grad_(True)
        self.unet(
            latents,
            t,
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=self.ca_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        self.unet.zero_grad()

        attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"))
        loss = self._compute_contrastive_loss(
            attention_maps=attn_maps,
            attention_maps_t_plus_one=attention_maps_t_plus_one,
            token_groups=token_groups,
            loss_type=loss_fn,
            do_smoothing=do_smoothing,
            temperature=temperature,
            smoothing_kernel_size=smoothing_kernel_size,
            smoothing_sigma=smoothing_sigma,
            softmax_normalize=softmax_normalize,
            softmax_normalize_attention_maps=softmax_normalize_attention_maps,
        )
        del attn_maps
        torch.cuda.empty_cache()
        return loss, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        token_groups: List[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_iter_to_alter: int = 25,
        refinement_steps: int = 20,
        iterative_refinement_steps: List[int] = [0, 10, 20],
        scale_factor: int = 20,
        attn_res: Optional[Tuple[int, int]] = (16, 16),
        steps_to_save_attention_maps: Optional[List[int]] = None,
        do_smoothing: bool = True,
        smoothing_kernel_size: int = 3,
        smoothing_sigma: float = 0.5,
        temperature: float = 0.5,
        softmax_normalize: bool = True,
        softmax_normalize_attention_maps: bool = False,
        add_previous_attention_maps: bool = True,
        previous_attention_map_anchor_step: Optional[int] = None,
        loss_fn: str = "ntxent",
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        text_inputs=None,
        target_prompt=None,
        **kwargs,
    ):
        # Handle deprecated callback signature
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` to `__call__` is deprecated; use `callback_on_step_end` instead.",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` to `__call__` is deprecated; use `callback_on_step_end` instead.",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        run_standard_sd = kwargs.get("run_standard_sd", False)
        self.ca_kwargs = cross_attention_kwargs

        # 0. Default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._denoising_end = kwargs.get("denoising_end", None)
        self._interrupt = False

        # 2. Determine batch size
        if prompt is not None:
            self.prompt = prompt
            prompt_batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            prompt_batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode prompt
        if prompt_embeds is None:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=negative_prompt_2 or prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=cross_attention_kwargs.get("scale", None) if cross_attention_kwargs else None,
                clip_skip=clip_skip,
            )

        # 4. Timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, kwargs.get("timesteps", None), kwargs.get("sigmas", None)
        )

        # 5. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            prompt_batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Extra kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare SDXL micro-conditioning
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                original_size=negative_original_size,
                crops_coords_top_left=negative_crops_coords_top_left,
                target_size=negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(prompt_batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                prompt_batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )

        # 8. Initialize attention store & register attention control
        if attn_res is None:
            attn_res = (int(np.ceil(width / 32)), int(np.ceil(height / 32)))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        max_iter_to_alter = max_iter_to_alter or (len(timesteps) + 1)

        # text_embeddings used in refinement steps (only text-conditioned part)
        text_embeddings = prompt_embeds[prompt_batch_size * num_images_per_prompt:] if do_classifier_free_guidance else prompt_embeds

        attention_map_record = [{} for _ in range(num_images_per_prompt)]
        attention_map_t_plus_one = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                # Prepare added_cond_kwargs for SDXL
                added_cond_kwargs = {
                    "text_embeds": torch.zeros_like(add_text_embeds[1:]),
                    "time_ids": add_time_ids[1:],
                }
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds[1:]  if do_classifier_free_guidance else image_embeds

                # Gradient-based contrastive refinement
                with torch.enable_grad():
                    latents = latents.detach().requires_grad_(True)
                    updated_latents = []
                    for j in range(prompt_batch_size * num_images_per_prompt):
                        latent = latents[j].unsqueeze(0)
                        grp = [token_groups] if isinstance(token_groups[0], int) else token_groups
                        text_emb = text_embeddings if not do_classifier_free_guidance else text_embeddings[j:j+1]

                        # Forward pass to store attention
                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_emb.unsqueeze(0) if text_emb.ndim == 2 else text_emb,
                            timestep_cond=None,
                            cross_attention_kwargs=self.ca_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        self.unet.zero_grad()

                        attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"))
                        if steps_to_save_attention_maps and i in steps_to_save_attention_maps:
                            attention_map_record[j // num_images_per_prompt][i] = attn_maps.detach().cpu()

                        loss = self._compute_contrastive_loss(
                            attention_maps=attn_maps,
                            attention_maps_t_plus_one=attention_map_t_plus_one,
                            token_groups=token_groups,
                            loss_type=loss_fn,
                            do_smoothing=do_smoothing,
                            temperature=temperature,
                            smoothing_kernel_size=smoothing_kernel_size,
                            smoothing_sigma=smoothing_sigma,
                            softmax_normalize=softmax_normalize,
                            softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                        )
                        del attn_maps
                        torch.cuda.empty_cache()

                        if i in iterative_refinement_steps:
                            loss, refined_latent = self._perform_iterative_refinement_step(
                                latents=latent,
                                token_groups=token_groups,
                                loss=loss,
                                text_embeddings=text_emb.unsqueeze(0) if text_emb.ndim == 2 else text_emb,
                                step_size=scale_factor * math.sqrt(scale_range[i]),
                                t=t,
                                refinement_steps=refinement_steps,
                                do_smoothing=do_smoothing,
                                smoothing_kernel_size=smoothing_kernel_size,
                                smoothing_sigma=smoothing_sigma,
                                temperature=temperature,
                                softmax_normalize=softmax_normalize,
                                softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                                attention_maps_t_plus_one=attention_map_t_plus_one,
                                loss_fn=loss_fn,
                                added_cond_kwargs=added_cond_kwargs
                            )
                            torch.cuda.empty_cache()
                            latent = refined_latent

                        if i < max_iter_to_alter and loss != 0:
                            latent = self._update_latent(
                                latents=latent,
                                loss=loss,
                                step_size=scale_factor * math.sqrt(scale_range[i]),
                            )
                        del loss
                        torch.cuda.empty_cache()
                        updated_latents.append(latent)

                    latents = torch.cat(updated_latents, dim=0)
                    del updated_latents
                    torch.cuda.empty_cache()

                if add_previous_attention_maps and (
                    previous_attention_map_anchor_step is None or i == previous_attention_map_anchor_step
                ):
                    attention_map_t_plus_one = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"))
                    torch.cuda.empty_cache()

                # Expand for classifier-free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise residual
                added_cond_for_denoise = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_for_denoise["image_embeds"] = image_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=self.ca_kwargs,
                    added_cond_kwargs=added_cond_for_denoise,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_text, guidance_rescale=guidance_rescale)

                # Denoise step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                del noise_pred, latent_model_input
                torch.cuda.empty_cache()
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                # Callbacks
                if callback_on_step_end is not None:
                    cb_kwargs = {}
                    for key in callback_on_step_end_tensor_inputs:
                        cb_kwargs[key] = locals()[key]
                    cb_outputs = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_outputs.get("latents", latents)
                    prompt_embeds = cb_outputs.get("prompt_embeds", prompt_embeds)
                    add_text_embeds = cb_outputs.get("add_text_embeds", add_text_embeds)
                    add_time_ids = cb_outputs.get("add_time_ids", add_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 10. Decode latents to images
        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype and torch.backends.mps.is_available():
                self.vae = self.vae.to(latents.dtype)

            # Denormalize latents
            has_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_mean and has_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # 11. Offload
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, attention_map_record)
        return StableDiffusionXLPipelineOutput(images=image), attention_map_record

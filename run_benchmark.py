import torch
import os
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline
from diffusers import PNDMScheduler

import configs.envs
from configs import benchmark
from methods.pipeline_attend_and_excite import AttendAndExcitePipeline
from methods.pipeline_conform import ConformXLPipeline
from methods.TEBOpt import TEBOptSDXL
from methods.dos import DOS, SDXLTextEncoder, SD3_5TextEncoder
from utils import ptp_utils
from utils.parser import get_object_indices
from utils.ptp_utils import AttentionStore
from configs.config import AEConfig, CONFORMConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="cuda(for nvidia) or mps(for apple sillicon)")
    parser.add_argument("--output_path", default="outputs/performance_comparison/sdxl")
    parser.add_argument("--dataset")
    parser.add_argument("--method")
    parser.add_argument("--seed_range", type=int, nargs=2, help="tuple input with two values. ex) 1 3")

    #### experimental arguments ####
    parser.add_argument("--lambda_sep", type=float, default=1.0, help="[experiment] only for ablation study")
    parser.add_argument("--disable_separating_object_embedding", action="store_true", help="[experiment] only for ablation study")
    parser.add_argument("--disable_separating_eot_embedding", action="store_true", help="[experiment] only for ablation study")
    parser.add_argument("--disable_separating_pooled_embedding", action="store_true", help="[experiment] only for ablation study")

    args = parser.parse_args()
    return args


def run_benchmark(args):
    pipe = load_pipeline(args)
    method = load_method(args)

    try:
        dataset = getattr(benchmark, args.dataset)
    except AttributeError:
        valid = [n for n in dir(benchmark) if not n.startswith("_")]
        raise ValueError(f"Invalid dataset {args.dataset!r}.  "
                         f"Available: {', '.join(valid)}")

    for target_prompt_and_objects in dataset:
        target_prompt = target_prompt_and_objects[0]
        target_objects = target_prompt_and_objects[1:]
        for seed in range(args.seed_range[0], args.seed_range[1]):
            target_dir = os.path.join(args.output_path, target_prompt)
            if os.path.isfile(os.path.join(target_dir, f"{seed}.png")): 
                print(f"{os.path.join(target_dir, f'{seed}.png')} already exists! skipping this prompt...")
                continue

            image = method(pipe, seed, target_prompt, target_objects, args)

            os.makedirs(target_dir, exist_ok=True)
            image.save(os.path.join(target_dir, f"{seed}.png"))


def load_method(args):
    if args.method in ["sdxl", "sd3.5"]:
        method = default
    elif args.method == "sdxl_with_tebopt":
        method = tebopt_sdxl
    elif args.method == "sdxl_with_attend_and_excite":
        method = attend_and_excite_sdxl
    elif args.method == "sdxl_with_conform":
        method = conform_sdxl
    elif args.method == "sdxl_with_dos":
        method = dos_sdxl
    elif args.method == "sd3.5_with_dos":
        method = dos_sd3_5
    else:
        raise ValueError("invalid method")
    return method


def load_pipeline(args):
    if args.method in ["sdxl_with_attend_and_excite"]:
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", set_alpha_to_one=False, skip_prk_steps=True, steps_offset=1)
        pipe = AttendAndExcitePipeline.from_pretrained(
            model_name, scheduler=scheduler, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(args.device)
    elif args.method in ["sdxl_with_conform"]:
        pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(args.device)
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", set_alpha_to_one=False, skip_prk_steps=True, steps_offset=1)
        pipe = ConformXLPipeline(
            vae=pipe_sdxl.vae,
            text_encoder=pipe_sdxl.text_encoder,
            tokenizer=pipe_sdxl.tokenizer,
            text_encoder_2=pipe_sdxl.text_encoder_2,
            tokenizer_2=pipe_sdxl.tokenizer_2,
            unet=pipe_sdxl.unet,
            scheduler=scheduler,
            feature_extractor=pipe_sdxl.feature_extractor,
            image_encoder=pipe_sdxl.image_encoder
        )
    elif args.method.split("_")[0] == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(args.device)
    elif args.method.split("_")[0] == "sd3.5":
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=args.device) # It is needed if you have less than 24GB gpu memories...
    else:
        raise ValueError("invalid pipeline")
    return pipe


def default(pipe, seed, target_prompt, target_objects, args):
    generator = torch.Generator(args.device).manual_seed(seed)
    image = pipe(prompt=[target_prompt], generator=generator).images[0]
    return image


def tebopt_sdxl(pipe, seed, target_prompt, target_objects, args):
    text_embeddings, _, pooled_text_embeddings, _ = pipe.encode_prompt([target_prompt])
    text_embeddings, pooled_text_embeddings = text_embeddings.clone().detach(), pooled_text_embeddings.clone().detach()
    embedding_optimizer = TEBOptSDXL(pipe=pipe, device=args.device)

    text_embeddings = embedding_optimizer.optimize_text_embeddings(
        target_prompt=target_prompt,
        target_objects=target_objects,
        text_embeddings=text_embeddings,
        lr = args.lr if hasattr(args, 'lr') else None
    )

    generator = torch.Generator(args.device).manual_seed(seed)
    image = pipe(prompt_embeds=text_embeddings, pooled_prompt_embeds=pooled_text_embeddings, generator=generator).images[0]
    return image


def attend_and_excite_sdxl(pipe, seed, target_prompt, target_objects, args):
    config = AEConfig(target_prompt)
    generator = torch.Generator(args.device).manual_seed(seed)
    controller = AttentionStore()
    ptp_utils.register_attention_control(pipe, controller)
    object_indices = get_object_indices(target_prompt, target_objects, pipe.tokenizer)
    image = pipe(prompt=target_prompt,
                 attention_store=controller,
                 indices_to_alter=[x[0] for x in object_indices],
                 attention_res=config.attention_res,
                 guidance_scale=config.guidance_scale,
                 generator=generator,
                 num_inference_steps=config.n_inference_steps,
                 max_iter_to_alter=config.max_iter_to_alter,
                 run_standard_sd=config.run_standard_sd,
                 thresholds=config.thresholds,
                 scale_factor=args.scale_factor if hasattr(args, 'scale_factor') else config.scale_factor,
                 scale_range=config.scale_range,
                 smooth_attentions=config.smooth_attentions,
                 sigma=config.sigma,
                 kernel_size=config.kernel_size).images[0]
    return image


def conform_sdxl(pipe, seed, target_prompt, target_objects, args):
    config = CONFORMConfig()
    generator = torch.Generator(args.device).manual_seed(seed)
    token_groups = get_object_indices(target_prompt, target_objects, pipe.tokenizer)
    images, _ = pipe(
        prompt=target_prompt,
        token_groups=token_groups,
        guidance_scale=config.guidance_scale,
        generator=generator,
        num_inference_steps=config.num_inference_steps,
        max_iter_to_alter=config.max_iter_to_alter,
        attn_res=config.attn_res,
        scale_factor=args.scale_factor if hasattr(args, 'scale_factor') else config.scale_factor,
        iterative_refinement_steps=config.iterative_refinement_steps,
        steps_to_save_attention_maps=config.steps_to_save_attention_maps,
        do_smoothing=config.do_smoothing,
        smoothing_sigma=config.smoothing_sigma,
        smoothing_kernel_size=config.smoothing_kernel_size,
        temperature=config.temperature,
        refinement_steps=config.refinement_steps,
        softmax_normalize=config.softmax_normalize,
        softmax_normalize_attention_maps=config.softmax_normalize_attention_maps,
        add_previous_attention_maps=config.add_previous_attention_maps,
        previous_attention_map_anchor_step=config.previous_attention_map_anchor_step,
        loss_fn=config.loss_fn,
    )
    return images.images[0]


def dos_sdxl(pipe, seed, target_prompt, target_objects, args):
    text_embeddings, _, pooled_text_embeddings, _ = pipe.encode_prompt([target_prompt])
    text_embeddings, pooled_text_embeddings = text_embeddings.clone().detach(), pooled_text_embeddings.clone().detach()
    text_encoder = SDXLTextEncoder(pipe=pipe)
    text_embeddings, pooled_text_embeddings = _dos(pipe, target_prompt, target_objects, text_embeddings, pooled_text_embeddings, args, text_encoder)

    generator = torch.Generator(args.device).manual_seed(seed)
    image = pipe(prompt_embeds=text_embeddings, pooled_prompt_embeds=pooled_text_embeddings, generator=generator).images[0]
    return image


def dos_sd3_5(pipe, seed, target_prompt, target_objects, args):
    text_embeddings, _, pooled_text_embeddings, _ = pipe.encode_prompt([target_prompt], None, None)
    text_embeddings, pooled_text_embeddings = text_embeddings.clone().detach(), pooled_text_embeddings.clone().detach()
    text_encoder = SD3_5TextEncoder(pipe=pipe)
    text_embeddings, pooled_text_embeddings = _dos(pipe, target_prompt, target_objects, text_embeddings, pooled_text_embeddings, args, text_encoder)

    generator = torch.Generator(args.device).manual_seed(seed)
    image = pipe(prompt_embeds=text_embeddings, pooled_prompt_embeds=pooled_text_embeddings, generator=generator).images[0]
    return image


def _dos(pipe, target_prompt, target_objects, text_embeddings, pooled_text_embeddings, args, text_encoder):
    embedding_calibrator = DOS(pipe=pipe, text_encoder=text_encoder, device=args.device)

    text_embeddings, pooled_text_embeddings = embedding_calibrator.update_text_embeddings(
        target_prompt=target_prompt,
        target_objects=target_objects,
        text_embeddings=text_embeddings, 
        pooled_text_embeddings=pooled_text_embeddings,
        separating_object_embedding=not args.disable_separating_object_embedding if hasattr(args, 'disable_separating_object_embedding') else True,
        separating_eot_embedding=not args.disable_separating_eot_embedding if hasattr(args, 'disable_separating_eot_embedding') else True,
        separating_pooled_embedding=not args.disable_separating_pooled_embedding if hasattr(args, 'disable_separating_pooled_embedding') else True,
        lambda_sep=args.lambda_sep
    )
    return text_embeddings, pooled_text_embeddings


if __name__=="__main__":
    args = parse_args()
    run_benchmark(args)

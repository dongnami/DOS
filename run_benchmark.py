import torch
import os
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline

import configs.envs
from configs import benchmark
from methods.dos import DOS, SDXLTextEncoder, SD3_5TextEncoder


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
    elif args.method == "sdxl_with_dos":
        method = dos_sdxl
    elif args.method == "sd3.5_with_dos":
        method = dos_sd3_5
    else:
        raise ValueError("invalid method")
    return method


def load_pipeline(args):
    if args.method.split("_")[0] == "sdxl":
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

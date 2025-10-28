import json
import os
import glob
import base64
import time
import re
import requests
import argparse
import concurrent.futures
import io
from PIL import Image

from requests import api


def encode_image_to_base64(img_path: str, size=(256,256), fmt="PNG"):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"cannot find image: {img_path}")
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.Resampling.BILINEAR)

        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_image


def parse_prompt_to_objects(prompt_string: str):
    parts = re.split(r",|\s+and\s+", prompt_string.lower())
    object_names = set()
    articles_numbers = re.compile(r"^(a|an|the|\d+)\s+")
    for part in parts:
        clean_part = part.strip()
        if clean_part:
            core_name = articles_numbers.sub("", clean_part).strip()
            if core_name:
                object_names.add(core_name)
    return object_names


def find_images_in_folder(folder_path: str):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))

    def sort_key(filepath):
        filename = os.path.basename(filepath)
        name_part = os.path.splitext(filename)[0]
        try:
            return int(name_part)
        except ValueError:
            return name_part

    files.sort(key=sort_key)
    return files


def get_prompt_from_image_path(image_path: str) -> str:
    return os.path.basename(os.path.dirname(image_path))


class VLMEvaluator:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_evaluation_folder: str,
        api_call_delay: float = 0.5,
        max_workers: int = 4,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.base_evaluation_folder = base_evaluation_folder
        self.max_workers = max_workers
        self.api_call_delay = api_call_delay

    def _call_vlm_api(self, image_path: str):
        base64_image = encode_image_to_base64(image_path)
        img_struct = dict(url=f"data:image/jpeg;base64,{base64_image}")

        prompt_string = get_prompt_from_image_path(image_path)

        vlm_task_prompt = f"""
        Analyze the image and for each of the following objects—{prompt_string}—determine whether it:

        1. appears **fully intact** — clearly depicted with correct shape, texture, and all native parts present  
        2. appears **mixed** — blended or merged in shape, texture, or any feature (e.g., scales, skin pattern, tail, fin) from another object  
        3. is **absent** — not present at all (neither fully intact nor mixed)

        ### Output Format
        Provide exactly N comma-separated tokens in this fixed order: 
        `[object1_status], [object2_status], ..., [objectN_status]`

        where each `[*_status]` must be:
        - the object name itself (e.g. `cat`) **only if** it appears **fully intact**  
        - `mixed` if it appears blended or mixed **in any way**
        - `absent` if it does not appear either fully intact or mixed

        Do **not** include any additional text or explanations.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="image_url", image_url=img_struct),
                    {"type": "text", "text": vlm_task_prompt},
                ],
            }
        ]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps({"model": self.model_id, "messages": messages}),
                timeout=60,
            )
            response.raise_for_status()  # HTTP error
            resp_struct = response.json()
            if "error" in resp_struct:
                print(
                    f"API Error for {os.path.basename(image_path)}: {resp_struct['error']}"
                )
                return ""
            return (
                resp_struct.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except requests.exceptions.Timeout:
            print(f"TIMEOUT_ERROR: {os.path.basename(image_path)}")
            return "TIMEOUT_ERROR"
        except requests.exceptions.RequestException as e:
            print(f"REQUEST_ERROR: ({os.path.basename(image_path)}): {e}")
            return "REQUEST_ERROR"
        except Exception as err:
            print(
                f"PROCESSING_ERROR: ({os.path.basename(image_path)}): {type(err)} - {err}"
            )
            return "PROCESSING_ERROR"

    def evaluate_single_image(self, image_path: str):
        prompt_string = get_prompt_from_image_path(image_path)
        expected_objects_from_prompt = parse_prompt_to_objects(prompt_string)
        vlm_response_raw = self._call_vlm_api(image_path)
        time.sleep(self.api_call_delay)
        vlm_response_lower = vlm_response_raw.lower()
        is_mixture = "mixed" in vlm_response_lower
        temp_object_list = []
        if not any(
            err_tag in vlm_response_raw
            for err_tag in ["TIMEOUT_ERROR", "REQUEST_ERROR", "PROCESSING_ERROR"]
        ):
            for obj_part in vlm_response_lower.split(","):
                obj_candidate = obj_part.strip()
                if obj_candidate and obj_candidate != "mixed":
                    temp_object_list.append(obj_candidate)

        vlm_detected_objects = set(temp_object_list)
        is_detection_success = expected_objects_from_prompt.issubset(
            vlm_detected_objects
        )
        detected_objects = ", ".join(sorted(list(vlm_detected_objects)))

        return {
            "image_path": image_path,
            "image_filename": os.path.basename(image_path),
            "prompt_string": prompt_string,
            "is_mixture": is_mixture,
            "is_detection_success": is_detection_success,
            "detected_objects": detected_objects,
            "vlm_raw_response": vlm_response_raw,
        }

    def run(self):
        all_image_paths_to_evaluate = []
        prompt_folders = [
            d
            for d in os.listdir(self.base_evaluation_folder)
            if os.path.isdir(os.path.join(self.base_evaluation_folder, d))
        ]

        for prompt_folder_name in prompt_folders:
            prompt_folder_path = os.path.join(
                self.base_evaluation_folder, prompt_folder_name
            )
            images_in_folder = find_images_in_folder(prompt_folder_path)
            if not images_in_folder:
                print(f"warning: there is no image in '{prompt_folder_name}'.")
                continue
            all_image_paths_to_evaluate.extend(images_in_folder)

        if not all_image_paths_to_evaluate:
            print("there is no image to evaluate.")
            return 0.0, 0.0, []

        all_eval_results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_img = {
                executor.submit(self.evaluate_single_image, img_path): img_path
                for img_path in all_image_paths_to_evaluate
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_img)):
                img_path = future_to_img[future]
                try:
                    result = future.result()
                    all_eval_results.append(result)
                    print(
                        f"  Processed ({i+1}/{len(all_image_paths_to_evaluate)}): {os.path.basename(img_path)}"
                    )
                except Exception as exc:
                    print(
                        f"Error raised (in processing - {os.path.basename(img_path)}): {exc}"
                    )
                    all_eval_results.append(
                        {
                            "image_path": img_path,
                            "image_filename": os.path.basename(img_path),
                            "prompt_string": get_prompt_from_image_path(img_path),
                            "is_mixture": False,
                            "is_detection_success": False,
                            "detected_objects": "EVALUATION_ERROR",
                            "vlm_raw_response": f"Error: {exc}",
                        }
                    )

        total_images = len(all_eval_results)
        if total_images == 0:
            return 0.0, 0.0, []

        successful_detections_count = sum(
            1 for res in all_eval_results if res["is_detection_success"]
        )
        mixture_detections_by_vlm_count = sum(
            1 for res in all_eval_results if res["is_mixture"]
        )
        overall_success_rate = (successful_detections_count / total_images) * 100
        overall_mixture_rate = (mixture_detections_by_vlm_count / total_images) * 100

        prompt_intermediate_results = {}
        for res in all_eval_results:
            prompt = res["prompt_string"]
            if prompt not in prompt_intermediate_results:
                prompt_intermediate_results[prompt] = {}
            prompt_intermediate_results[prompt][res["image_filename"]] = res[
                "vlm_raw_response"
            ]

        print(f"\n--- Evaluation Summary ---")
        print(f"The number of total images: {total_images}")
        print(f"Success Rate: {overall_success_rate:.2f}%")
        print(f"Mixture Rate: {overall_mixture_rate:.2f}%")

        results_filename = "vlm_evaluation_summary_results.json"
        output_dir = self.base_evaluation_folder.replace("outputs", "outputs_vlm")
        os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "overall_success_rate_percent": overall_success_rate,
            "overall_mixture_rate_percent": overall_mixture_rate,
            "overall_results_per_prompt": prompt_intermediate_results,
        }
        with open(os.path.join(output_dir, results_filename), "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"The evaluation results are saved in {os.path.join(output_dir, results_filename)}.")
        return overall_success_rate, overall_mixture_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM Evaluation Script for Object Detection and Mixture"
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Path to the base folder containing prompt subfolders with images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model ID for the VLM API (e.g., from OpenRouter).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenRouter API key. Recommended: Set OPENROUTER_API_KEY environment variable instead.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for API calls.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls within each worker thread.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="vlm_evaluation_results.json",
        help="Filename for the output JSON results.",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if args.api_key:
        api_key = args.api_key

    if not api_key:
        print(
            "Error: there is no OpenRouter API Key"
        )
        exit(1)

    if not os.path.isdir(args.folder):
        print(f"Error: there is no directory: {args.folder}")
        exit(1)
    try:
        evaluator = VLMEvaluator(
            model_id=args.model,
            api_key=api_key,
            base_evaluation_folder=args.folder,
            api_call_delay=args.delay,
            max_workers=args.workers,
        )
        print(f"\nStart evaluation:")
        print(f"  folder: {args.folder}")
        print(f"  model: {args.model}")
        print(f"  num workers: {args.workers}")
        print(f"  API delay: {args.delay} seconds")
        print(f"  output path: {args.output}\n")

        success_rate, mixture_rate = evaluator.run()

        print("\n--- Evaluation Summary ---")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Mixture Rate: {mixture_rate:.2f}%")
        print("--- Complete ---")

    except ValueError as ve:
        print(f"Value Error: {ve}")
        exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        exit(1)

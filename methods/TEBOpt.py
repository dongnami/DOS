import abc
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
import configs.envs
import io
import itertools
import torch.nn.functional as F
from PIL import Image
from transformers.image_utils import load_image
from itertools import permutations, combinations
from utils.parser import get_word_index, get_object_indices


class AbstractTEBOpt:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    @abc.abstractmethod
    def optimize_text_embeddings(self, target_prompt, object_indices):
        pass


class TEBOptSDXL(AbstractTEBOpt):

    def __init__(self, pipe, device):
        super().__init__(pipe, device)

    def optimize_text_embeddings(self, target_prompt, target_objects, text_embeddings, lr=None):
        if lr is None:
            lr = 0.01
        if type(target_prompt) == list:
            target_prompt = target_prompt[0]

        print("====================== start optimizing =========================")
        print(f"target prompt: '{target_prompt}'")
        for idx, target_prompt_object in enumerate(target_objects):
            print(f"target_prompt_obj[{idx}]: '{target_prompt_object}'")

        if len(target_objects) < 2:
            raise ValueError("The number of object indices is < 2")
        
        # 1) 오브젝트 토큰 인덱스 구하기
        object_indices = get_object_indices(target_prompt, target_objects, self.pipe.tokenizer)
        object_indices_flatttened = list(itertools.chain.from_iterable(object_indices))
        m = len(self.pipe.tokenizer.encode(target_prompt)) - 1
        k = len(target_objects)

        # 2) text_embeddings를 nn.Parameter로 감싸 Optimizer에 넣기
        #    전체 텐저가 Parameter지만, 업데이트는 아래 mask로 제한할 예정
        text_embeddings_param = torch.nn.Parameter(text_embeddings.float(), requires_grad=True)
        optimizer = torch.optim.Adam([text_embeddings_param], lr=lr)

        # 3) "a photo of a {obj}"에 대한 임베딩을 미리 한 번만 계산
        obj_pure_embeddings_list = []
        for obj in target_objects:
            prompt_obj_pure = [f"a photo of a {obj}"]
            text_embeddings_obj_pure, _, _, _ = self.pipe.encode_prompt(prompt_obj_pure)[:4]
            obj_pure_token_indices = get_word_index(prompt_obj_pure[0], obj, self.pipe.tokenizer)
            obj_pure_emb = text_embeddings_obj_pure[0, obj_pure_token_indices].clone().detach()
            averaged_obj_pure_embedding = torch.mean(obj_pure_emb, dim=0)
            obj_pure_embeddings_list.append(averaged_obj_pure_embedding)

        # --------------------- 학습 루프 시작 ---------------------
        for step in range(20):
            optimizer.zero_grad()
            positive_loss = 0.0
            negative_loss = 0.0

            # positive loss 계산
            similarity_between_obj_and_obj_pure_list = []
            for obj_idx, obj in enumerate(target_objects):
                obj_token_indices = object_indices[obj_idx]
                averaged_obj_pure_embedding = obj_pure_embeddings_list[obj_idx]
                for obj_token_idx_i in obj_token_indices:
                    sim = F.cosine_similarity(
                        text_embeddings_param[0, obj_token_idx_i],
                        averaged_obj_pure_embedding,
                        dim=0
                    )
                    similarity_between_obj_and_obj_pure_list.append(sim)
            positive_loss = torch.min(torch.stack(similarity_between_obj_and_obj_pure_list))

            # negative loss 계산 (현재 프롬프트의 다른 토큰들과의 유사도)
            for obj_idx, obj in enumerate(target_objects):
                obj_token_indices = object_indices[obj_idx]
                for obj_token_idx_i in obj_token_indices:
                    for obj_token_idx_j in range(1, m):
                        if obj_token_idx_j in obj_token_indices:
                            continue
                        negative_loss = negative_loss + F.cosine_similarity(
                            text_embeddings_param[0, obj_token_idx_i],
                            text_embeddings_param[0, obj_token_idx_j],
                            dim=0
                        )
            negative_loss = negative_loss / (k * (m - 1))

            loss = -positive_loss + negative_loss
            loss.backward()

            optimizer.step()

            if step % 4 == 0:
                print(f"Step {step}: Loss {loss.item():.4f} | "
                    f"Positive {-positive_loss.item():.4f} | Negative {negative_loss.item():.4f}")

            if loss.item() <= -0.7 and -positive_loss.item() <= -0.95 and negative_loss.item() <=0.25:
                break
        
        final_text_embeddings = text_embeddings_param.data.half()
        return final_text_embeddings.clone().detach()

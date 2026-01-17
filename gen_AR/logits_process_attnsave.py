from transformers.generation import LogitsProcessorList, LogitsProcessor 
import inspect
import math
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

import os


class BatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.special_token_ids = torch.tensor([151643, 151849, 151850, 151851, 151852, 151853, 151846, 151847])

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
                
            next_flag = torch.full_like(input_ids, -1)
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits
    
    def __call__(self, input_ids, scores):

        #scores = torch.nn.functional.log_softmax(scores, dim=-1)
        cond_logits = scores[0, :]
        uncond_logits = scores[1, :]
        variance_cond = cond_logits.pow(2).mean(-1, keepdim=True)
        variance_uncond = uncond_logits.pow(2).mean(-1, keepdim=True)
        eps = 1e-12  # to avoid division by zero

        # scale factor per batch element: sqrt(var_cond / var_uncond)
        scale = torch.sqrt(variance_cond / (variance_uncond + eps))  # shape: (B, 1)

        unconditional_logits_scaled = uncond_logits * scale
        
        scores_processed = 7 * (cond_logits - unconditional_logits_scaled) + unconditional_logits_scaled
        
        norm_scores_processed, _ = self.model.model.norm(scores_processed)
        
        scores_processed = self.model.lm_head(norm_scores_processed.to(self.model.dtype))
        
        scores = torch.stack([scores_processed, scores_processed], dim=0)
        
        return scores
        



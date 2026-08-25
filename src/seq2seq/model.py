"""
Conditioned Seq2Seq Transformer Model for Rosetta AI Code Translation.

Wraps a pre-trained CodeT5/T5 encoder-decoder transformer (`Salesforce/codet5-base` or `t5-small`)
with a soft-prompt projection layer that conditions generation on the 128-dim fused semantic representation vector.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL_NAME = "t5-small"



class ConditionedSeq2SeqModel(nn.Module):
    """
    Seq2Seq Transformer Model conditioned on Fused Semantic IR Embeddings.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, fused_dim: int = 128):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        d_model = self.seq2seq_model.config.d_model
        # Soft prompt projection layer projecting fused_dim -> d_model
        self.prompt_projection = nn.Linear(fused_dim, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fused_embed: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Forward pass with soft prompt conditioning prepended to encoder inputs.
        """
        inputs_embeds = self.seq2seq_model.encoder.embed_tokens(input_ids)  # (batch, seq_len, d_model)
        
        # Project fused vector to soft prompt token embedding
        soft_prompt = self.prompt_projection(fused_embed).unsqueeze(1)  # (batch, 1, d_model)
        
        # Prepend soft prompt token embedding to input embeddings
        conditioned_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        
        # Adjust attention mask for soft prompt token
        prompt_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        conditioned_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.seq2seq_model(
            inputs_embeds=conditioned_embeds,
            attention_mask=conditioned_mask,
            labels=labels
        )
        return outputs

    def generate_code(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str,
        fused_vec: List[float],
        max_length: int = 256,
        device: str = "cpu"
    ) -> str:
        """
        Generates translated target code for source snippet conditioned on fused representation.
        """
        self.eval()
        prompt_text = f"translate {source_lang} to {target_lang}: {source_code}"
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        fused_tensor = torch.tensor(fused_vec, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            inputs_embeds = self.seq2seq_model.encoder.embed_tokens(inputs["input_ids"])
            soft_prompt = self.prompt_projection(fused_tensor).unsqueeze(1)
            conditioned_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)

            prompt_mask = torch.ones((inputs["attention_mask"].size(0), 1), device=device)
            conditioned_mask = torch.cat([prompt_mask, inputs["attention_mask"]], dim=1)

            generated_ids = self.seq2seq_model.generate(
                inputs_embeds=conditioned_embeds,
                attention_mask=conditioned_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        target_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return target_code


if __name__ == "__main__":
    model = ConditionedSeq2SeqModel("t5-small")
    print("ConditionedSeq2SeqModel initialized successfully!")

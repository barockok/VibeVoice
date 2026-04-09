"""Qwen2.5-7B-Instruct response generator for conversation demo."""

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo.conversation.response_generator import ResponseGenerator

SYSTEM_PROMPT = (
    "You are VibeVoice, a friendly and helpful voice AI assistant. "
    "Keep responses concise and conversational (1-3 sentences). "
    "You are speaking out loud, so avoid markdown, lists, or code blocks."
)


class QwenResponseGenerator(ResponseGenerator):
    """Response generator backed by Qwen2.5-7B-Instruct."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 150,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

    def generate(self, transcription: str, history: List[Dict[str, str]]) -> str:
        """Generate a response given the user transcription and conversation history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include last 10 turns from history
        # History entries use {"role": ..., "text": ...} but chat template needs "content"
        for turn in history[-10:]:
            messages.append({"role": turn["role"], "content": turn["text"]})

        # Append current user message
        messages.append({"role": "user", "content": transcription})

        # Apply chat template to get properly formatted input
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.model.device
        )
        input_length = input_ids.shape[1]

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

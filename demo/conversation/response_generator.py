"""Response generator interface for conversation demo."""

from typing import List, Dict


class ResponseGenerator:
    def generate(self, transcription: str, history: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class EchoResponseGenerator(ResponseGenerator):
    def generate(self, transcription: str, history: List[Dict[str, str]]) -> str:
        return f"You said: {transcription}"


class TemplateResponseGenerator(ResponseGenerator):
    RESPONSES = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What would you like to talk about?",
        "how are you": "I'm doing great, thanks for asking! How about you?",
        "what is your name": "I'm VibeVoice, a real-time voice AI assistant.",
        "goodbye": "Goodbye! It was nice talking to you.",
        "bye": "See you later! Have a great day.",
        "thank you": "You're welcome! Is there anything else I can help with?",
        "thanks": "Happy to help! Anything else?",
    }

    def generate(self, transcription: str, history: List[Dict[str, str]]) -> str:
        lower = transcription.lower().strip().rstrip(".,!?")
        for key, response in self.RESPONSES.items():
            if key in lower:
                return response
        return f"I heard you say: {transcription}. That's interesting! Tell me more."

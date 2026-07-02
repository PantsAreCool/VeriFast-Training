import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """Base Class defining the contract for LLM providers."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """
        Generates a textual response given a list of chat messages
        
        Expected message format:
        [
            {"role": "system", "content": "You are an expert sysadmin..."},
            {"role": "user", "content": "Fix this error..."}
        ]
        """
        pass


class OpenAIProvider(LLMInterface):
    """Wrapper for OpenAI's API SDK."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        self.model = model


    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMInterface):
    """Wrapper for Anthropic's SDK."""
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20240620"):

        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        self.model = model


    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        # Anthropic splits the system prompt out from the message history array ? Weird
        system_prompt = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.messages.create(model=self.model, max_tokens=1024, temperature=temperature, system=system_prompt, messages=user_messages)
        return response.content[0].text.strip()
    




def get_llm_client(provider_name: str, **kwargs) -> LLMInterface:
    provider_name = provider_name.lower().strip()
    
    if provider_name == "openai":
         return OpenAIProvider(**kwargs)
        
    elif provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
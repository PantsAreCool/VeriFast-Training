class ModelConfig:
    def __init__(self, model, temperature, max_tokens):
        self.allowed_models = ["gpt-4", "claude-sonnet", "gemini-pro"]
        
        if model not in self.allowed_models:
            raise ValueError(f"Invalid model '{model}'. Choices are: {self.allowed_models}")
            
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"Temperature {temperature} is out of bounds. Must be between 0.0 and 2.0")
            
        if max_tokens <= 0:
            raise ValueError(f"Max tokens {max_tokens} must be a positive integer value")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"ModelConfig(model='{self.model}', temperature={self.temperature}, max_tokens={self.max_tokens})"


if __name__ == "__main__":
    valid_config = ModelConfig(model="gpt-4", temperature=0.7, max_tokens=2048)
    print(f"Valid Configuration: {valid_config}")
    
    try:
        invalid_config = ModelConfig(model="unknown-llm", temperature=0.7, max_tokens=100)
    except ValueError as e:
        print(f"Caught Expected Error (Model): {e}")

    try:
        invalid_config = ModelConfig(model="gpt-4", temperature=2.5, max_tokens=100)
    except ValueError as e:
        print(f"Caught Expected Error (Temperature): {e}")

    try:
        invalid_config = ModelConfig(model="gpt-4", temperature=0.7, max_tokens=-50)
    except ValueError as e:
        print(f"Caught Expected Error (Tokens): {e}")
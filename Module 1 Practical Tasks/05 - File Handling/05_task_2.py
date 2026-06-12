import json
from pathlib import Path

class ConfigManager:
    def __init__(self, json_path="config.json", env_path=".env"):
        self.json_path = Path(json_path)
        self.env_path = Path(env_path)
        self.config = {}

    def load_env(self):
        env_vars = {}
        if not self.env_path.exists():
            return env_vars

        with open(self.env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip().strip('"').strip("'")
        return env_vars

    def initialize_config(self, required_keys=[]):
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            print(f"Base JSON file '{self.json_path}' not found. Starting empty.")
            self.config = {}

        env_vars = self.load_env()
        for key, value in env_vars.items():
            self.config[key] = value

        missing = []
        for key in required_keys:
            if key not in self.config:
                missing.append(key)
        
        if missing:
            raise ValueError(f"Configuration missing mandatory keys: {missing}")

    def get(self, key, default=None):
        return self.config.get(key, default)



if __name__ == "__main__":
    Path("config.json").write_text('{"model": "gpt-4", "temperature": 0.7}')
    Path(".env").write_text('temperature=0.2\nAPI_KEY=secret123')

    cfg = ConfigManager()
    cfg.initialize_config(required_keys=["model", "API_KEY"])

    print(f"Model (from JSON): {cfg.get('model')}")
    print(f"Temperature (overridden by .env): {cfg.get('temperature')}")
    print(f"API_KEY (from .env): {cfg.get('API_KEY')}")
    print(f"Max Tokens (default value fallback): {cfg.get('max_tokens', 2048)}")

    Path("config.json").unlink()
    Path(".env").unlink()
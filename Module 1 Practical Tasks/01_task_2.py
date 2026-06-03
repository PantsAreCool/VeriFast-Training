# Task 2
llm_database = {
    "llm1": {"provider": "provider1", "temperature": 0.8, "max_tokens": 2048},
    "llm2": {"provider": "provider2", "temperature": 0.9, "max_tokens": 4096},
    "llm3": {"provider": "provider3", "temperature": 0.5, "max_tokens": 8192},
    "llm4": {"provider": "provider4", "temperature": 0.6, "max_tokens": 512},
    "llm5": {"provider": "provider5", "temperature": 0.7, "max_tokens": 1024}
}


def search_by_provider(provider_name):
    print(f"\nModels by Provider: {provider_name}")
    found = False
    for model_name, info in llm_database.items():
        if info["provider"].lower() == provider_name.lower():
            print(f"{model_name} (Temp: {info['temperature']}, Max Tokens: {info['max_tokens']})")
            found = True
    
    if found != True:
        print("No models found for this provider.")


def filter_by_max_tokens(min_tokens):
    print(f"\nModels with Max Tokens >= {min_tokens}")
    found = False
    for model_name, info in llm_database.items():
        if info["max_tokens"] >= min_tokens:
            print(f"{model_name}: capacity of {info['max_tokens']} tokens")
            found = True
    
    if found != True:
        print("No models found for this provider.")


def compare_temperatures():
    print("\nTemperature Comparison Matrix")
    for model_name, info in llm_database.items():
        print(f"{model_name} | Temp: {info['temperature']}")


# Test
if __name__ == "__main__":
    search_by_provider("provider1")
    search_by_provider("provider2")
    search_by_provider("provider6")

    filter_by_max_tokens(4000)
    filter_by_max_tokens(16384)

    compare_temperatures()
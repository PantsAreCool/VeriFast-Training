# Task 1: Token Counting and Cost Analysis
# Choose 5 different text samples: a tweet (under 280 chars), a news article paragraph (~500 words), a Python code file (~100 lines), 
# a technical paper abstract (~200 words), and a page of creative writing (~1000 words). 
# Using tiktoken with the cl100k_base encoding, count the tokens for each sample. Calculate the characters-per-token ratio for each. 
# Then estimate the cost to process each sample through GPT-4o (input only) and through the embedding API. Present your results in a formatted table.

import tiktoken

def analyze_tokens_and_costs():
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return
    GPT4O_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
    EMBEDDING_COST_PER_TOKEN = 0.02 / 1_000_000

    samples = {
        "1. Tweet (dril)": (
            "Food $200. Data $150. Rent $800. Candles $3,600. Utility $150. "
            "someone who is good at the economy please help me budget this. "
            "my family is dying."
        ),
        "2. News Article (BBC)": (
            "Critics have given Sally Rooney's fourth novel, Intermezzo, positive "
            "reviews, with one calling it 'utterly perfect'. The book follows two "
            "brothers - Peter, a successful 30-something Dublin lawyer, and Ivan, "
            "a 22-year-old competitive chess player - who are grieving the recent "
            "death of their father."
        ),
        "3. Python Code": (
            'def analyze_vocabulary(text):\n'
            '    cleaned = text.lower()\n'
            '    for char in ".,!?;:\'\\"()[]{}\\n":\n'
            '        cleaned = cleaned.replace(char, " ")\n'
            '    tokens = cleaned.split()\n'
            '    if not tokens:\n'
            '        return\n'
            '    total_tokens = len(tokens)\n'
            '    unique_tokens = set(tokens)\n'
            '    vocab_richness = len(unique_tokens) / total_tokens'
        ),
        "4. Tech Paper Abstract": (
            "Large language models have shown remarkable capabilities across various "
            "natural language processing tasks. However, their internal alignment mechanics "
            "and optimization trajectories remain poorly understood. In this paper, we present "
            "a systematic empirical investigation into the token-level optimization landscapes "
            "of transformer-based architectures. By tracking gradient updates across millions "
            "of parameters, we demonstrate that structural alignment phase transitions heavily "
            "influence down-stream factual generalization."
        ),
        "5. Creative Writing (Doctor Who)": (
            "As you come into this world, something else is also born. You begin your life, "
            "and it begins a journey towards you. It moves never faster, never slower, perhaps "
            "a thousand miles away, but it never stops. It never hurries, it never turns aside. "
            "From that moment on, if you look in a mirror, you will see a shadow behind you. "
            "Only the Second Click. Only the clock ticking. And when you arrive at your "
            "destination, it will be standing there waiting. You can outrun it, you can trick "
            "it, you can hide, but one day, you will be in a room and there will be no way out, "
            "no doors, no windows. And it will walk in. And you will understand. "
            "Clara, my Clara, I will always remember you. The stars are going out, the universe "
            "is turning to dust, but this room stays. The wall is twelve feet pure diamond. "
            "Harder than anything else in the cosmos. And I am going to break it down with my "
            "bare fists. If it takes me a day, a year, or ten billion years, I will get through."
        )
    }

    header_format = "{:<25} | {:<10} | {:<12} | {:<10} | {:<18} | {:<18}"
    row_format = "{:<25} | {:<10,} | {:<12,} | {:<10.2f} | ${:<17.7f} | ${:<17.7f}"
    
    print("-" * 107)
    print(header_format.format("Sample Type", "Chars", "Tokens", "Char/Token", "GPT-4o Cost", "Embedding Cost"))
    print("-" * 107)

    for label, text in samples.items():
        char_count = len(text)
        token_count = len(encoder.encode(text))
        
        char_per_token = char_count / token_count if token_count > 0 else 0
        gpt4o_cost = token_count * GPT4O_INPUT_COST_PER_TOKEN
        embedding_cost = token_count * EMBEDDING_COST_PER_TOKEN
        
        print(row_format.format(label, char_count, token_count, char_per_token, gpt4o_cost, embedding_cost))
    
    print("-" * 107)

if __name__ == "__main__":
    analyze_tokens_and_costs()
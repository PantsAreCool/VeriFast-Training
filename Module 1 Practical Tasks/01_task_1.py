# Task 1
def analyze_vocabulary(text):
    cleaned = text.lower()
    for char in ".,!?;:'\"()[]{}\n":
        cleaned = cleaned.replace(char, " ")
    tokens = cleaned.split()
    
    if not tokens:
        print("No words found in the text.")
        return

    total_tokens = len(tokens)
    unique_tokens = set(tokens)
    vocab_richness = len(unique_tokens) / total_tokens
    
    total_length = sum(len(word) for word in tokens)
    avg_word_length = total_length / total_tokens

    word_counts = {}
    for i in tokens:
        word_counts[i] = word_counts.get(i, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    

    print("VOCABULARY ANALYSIS:")
    print(f"Total Tokens: {total_tokens}")
    print(f"Unique Tokens: {len(unique_tokens)}")
    print(f"Vocabulary Richness: {vocab_richness:.2f}")
    print(f"Avg Word Length: {avg_word_length:.1f} characters")
    print("Top 10 Most Common Words:")
    for rank, (word, freq) in enumerate(sorted_words[:10], 1):
        print(f"  {rank}. '{word}': {freq}x")

if __name__ == "__main__":
    sample = """
    Machine learning is a subset of artificial intelligence.
    Machine learning algorithms learn from data.
    Deep learning is a subset of machine learning.
    """
    analyze_vocabulary(sample)
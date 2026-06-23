# Tasj 3
import string

stop_words = {"the", "a", "an", "and", "is", "of", "to", "in", "on", "with"}


def preprocess_text(raw_text):
    lowercase_text = raw_text.lower()
    
    cleaned_text = ""
    for char in lowercase_text:
        if char not in string.punctuation:
            cleaned_text += char

    all_tokens = cleaned_text.split()
    
    final_tokens = []
    for token in all_tokens:
        if token not in stop_words:
            final_tokens.append(token)
            
    return final_tokens


# Testing
if __name__ == "__main__":
    
    line_1 = "The quick brown fox jumps over a lazy dog on a sunny day."
    line_2 = "Machine learning is a subset of artificial intelligence."
    line_3 = "Deep learning is a subset of machine learning."
    
    test_lines = [line_1, line_2, line_3]

    for i, line in enumerate(test_lines, start=1):
        cleaned_result = preprocess_text(line)
        
        print(f"\nParagraph {i}:")
        print(f"Original: {line}")
        print(f"Cleaned: {cleaned_result}")
        print(f"Word Count: Reduced from {len(line.split())} to {len(cleaned_result)} tokens.")
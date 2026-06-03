def run_pipeline(initial_input, list_of_functions):
    current_value = initial_input
    for func in list_of_functions:
        current_value = func(current_value)
    return current_value

def clean_text(text):
    cleaned = text.lower()
    for char in ".,!?;:'\"()[]{}\n":
        cleaned = cleaned.replace(char, " ")
    return cleaned

def tokenize(text):
    return text.split()

def count_words(tokens):
    counts = {}
    for word in tokens:
        counts[word] = counts.get(word, 0) + 1
    return counts

if __name__ == "__main__":
    raw_document = "Machine learning is a subset of artificial intelligence."
    pipeline_steps = [clean_text, tokenize, count_words]
    pipeline_result = run_pipeline(raw_document, pipeline_steps)
    print("Word Count Output:")
    print(pipeline_result)
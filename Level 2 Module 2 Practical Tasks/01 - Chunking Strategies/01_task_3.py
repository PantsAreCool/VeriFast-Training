# Task 3: Compare Chunking Strategies with Embedding Simulated Retrieval
# Write a function that, given a query string and a list of chunks, simulates retrieval by finding chunks with the highest word overlap with the query. 
# Compare how different chunking strategies (fixed-size 300, fixed-size 800, sentence-based, recursive) 
# affect the "retrieved" results for 3 different queries against the sample documents.

import re

def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def sentence_chunk(text: str, sentences_per_chunk: int = 3) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunks.append(" ".join(sentences[i:i + sentences_per_chunk]))
    return chunks

def recursive_chunk(text: str, chunk_size: int = 500) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split(sub_text: str, seps: list[str]) -> list[str]:
        if not sub_text: return []
        sep = seps[0]
        remaining = seps[1:]
        if sep == "":
            return [sub_text[i:i + chunk_size] for i in range(0, len(sub_text), chunk_size)]
        
        parts = sub_text.split(sep)
        chunks, current = [], ""
        for part in parts:
            part = part.strip()
            if not part: continue
            candidate = (current + " " + part).strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current: chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, remaining))
                    current = ""
                else:
                    current = part
        if current: chunks.append(current)
        return chunks

    return _split(text, separators)



def simulate_retrieval(query: str, chunks: list[str], top_k: int = 1) -> list[tuple[str, float]]:
    """
    Simulates vector retrieval using Jaccard Similarity word-overlap.
    """
    def get_words(text: str) -> set[str]:
        return set(re.findall(r'\b\w+\b', text.lower()))

    query_words = get_words(query)
    if not query_words:
        return []

    scored_chunks = []
    for chunk in chunks:
        chunk_words = get_words(chunk)
        if not chunk_words:
            continue
        intersection = query_words.intersection(chunk_words)
        union = query_words.union(chunk_words)
        similarity = len(intersection) / len(union)
        scored_chunks.append((chunk, similarity))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]


def run_retrieval_comparison():
    knowledge_base = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence.
    AI applications include advanced web search engines (e.g., Google Search), recommendation systems used by YouTube, Amazon, and Netflix, understanding human speech (e.g., Siri and Alexa), self-driving cars (e.g., Waymo), generative and creative tools (e.g., ChatGPT and AI art).
    
    Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. Common supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks.
    
    Unsupervised learning algorithms find structure in data, like grouping or clustering points, without labeled targets. Common algorithms include k-means clustering and autoencoders.
    
    Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward. The agent interacts with the environment by producing actions and getting feedback in the form of rewards or penalties.
    """.strip()

    queries = [
        "What applications use artificial intelligence like search engines or cars?",
        "How do supervised learning algorithms build models with inputs and outputs?",
        "What feedback does a reinforcement learning agent receive from its environment?"
    ]

    strategies = {
        "Fixed-Size (300 chars)": lambda t: fixed_size_chunk(t, chunk_size=300, overlap=30),
        "Fixed-Size (800 chars)": lambda t: fixed_size_chunk(t, chunk_size=800, overlap=80),
        "Sentence-Based (3 sent)": lambda t: sentence_chunk(t, sentences_per_chunk=3),
        "Recursive (400 chars)": lambda t: recursive_chunk(t, chunk_size=400)
    }

    for q_idx, query in enumerate(queries, 1):
        print(f"\n{'='*100}")
        print(f"QUERY {q_idx}: \"{query}\"")
        print(f"{'='*100}")
        
        for name, chunk_fn in strategies.items():
            chunks = chunk_fn(knowledge_base)
            results = simulate_retrieval(query, chunks, top_k=1)
            
            if results:
                top_chunk, score = results[0]
                display_chunk = top_chunk.replace('\n', ' ').strip()
                if len(display_chunk) > 120:
                    display_chunk = display_chunk[:117] + "..."
                
                print(f"-> {name:<25} | Score: {score:.3f} | Best Chunk: {display_chunk!r}")
            else:
                print(f"-> {name:<25} | Score: 0.000 | No Match Found")

if __name__ == "__main__":
    run_retrieval_comparison()
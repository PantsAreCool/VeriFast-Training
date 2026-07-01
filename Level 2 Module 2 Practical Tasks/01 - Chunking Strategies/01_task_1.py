# Task 1: Implement a Custom Token-Aware Chunker
# Build a chunking function that splits text based on approximate token count (roughly 4 characters per token for English). 
# It should accept a max_tokens parameter and produce chunks that stay within that token budget. 
# Test it with max_tokens=100 and max_tokens=256 on the sample documents.


import re

def token_aware_chunk(text: str, max_tokens: int = 100, overlap_tokens: int = 20) -> list[str]:
    """
    Splits text into chunks based on an approximate token budget.
    Assumes an average of 4 characters per token for English text.
    

    """
    CHARS_PER_TOKEN = 4
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _recursive_split(sub_text: str, seps: list[str]) -> list[str]:
        sub_text = sub_text.strip()
        if not sub_text:
            return []
            
        if len(sub_text) <= max_chars:
            return [sub_text]
            
        sep = seps[0]
        remaining_seps = seps[1:] if len(seps) > 1 else [""]
        
        if sep == "":
            chunks = []
            start = 0
            while start < len(sub_text):
                end = start + max_chars
                chunks.append(sub_text[start:end])
                start += max_chars - overlap_chars
            return chunks
            
        parts = sub_text.split(sep)
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            join_str = sep if sep in [". ", "\n\n", "\n"] else " "
            candidate = f"{current_chunk}{join_str}{part}" if current_chunk else part
            
            if len(candidate) <= max_chars:
                current_chunk = candidate
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    
                    overlap_text = current_chunk[-overlap_chars:] if overlap_chars > 0 else ""
                    if " " in overlap_text and sep != "":
                        overlap_text = overlap_text[overlap_text.find(" "):].strip()
                    current_chunk = f"{overlap_text}{join_str}{part}" if overlap_text else part
                    
                    if len(current_chunk) > max_chars:
                        if current_chunk != part:
                            current_chunk = part
                        if len(current_chunk) > max_chars:
                            final_chunks.extend(_recursive_split(current_chunk, remaining_seps))
                            current_chunk = ""
                else:
                    final_chunks.extend(_recursive_split(part, remaining_seps))
                    
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks

    return _recursive_split(text, separators)



# Test Verification
if __name__ == "__main__":
    sample_text = """
    Machine learning is a subset of artificial intelligence.
    Machine learning algorithms learn from data.
    Deep learning is a subset of machine learning.
    """.strip()

    for budget in [100, 256]:
        print(f"\nTesting Budget: {budget} tokens (~{budget * 4} chars)")
        chunks = token_aware_chunk(sample_text, max_tokens=budget, overlap_tokens=15)
        
        for idx, chunk in enumerate(chunks):
            approx_tokens = len(chunk) / 4
            print(f"Chunk {idx + 1} | Length: {len(chunk)} chars (~{approx_tokens:.1f} tokens)")
            print(f"Content: {chunk!r}\n{'-'*40}")
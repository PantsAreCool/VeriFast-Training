# Task 2: Build a Markdown Section Summarizer
# Using the markdown_header_chunk function, write a script that splits a long markdown document by headers, 
# and for each section prints the header title, the number of words, and the first sentence as a summary. 
# Test on a markdown file with at least 4 sections.

import re

def markdown_chunk(text: str, max_chunk_size: int = 2000) -> list[dict]:
    """
    Splits markdown text by headers, preserving header titles and hierarchy.
    """
    sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for section in sections:
        header_match = re.match(r'^(#{1,6})\s+(.+)', section)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).split('\n')[0].strip()
        else:
            level = 0
            title = "Introduction"

        chunks.append({
            "content": section,
            "header": title,
            "level": level
        })
    return chunks

def summarize_markdown_sections(markdown_text: str) -> None:
    """
    Parses a markdown document by headers and prints an executive summary for each section.
    """
    sections = markdown_chunk(markdown_text)
    
    print(f"\n{'='*80}")
    print(f"{'MARKDOWN DOCUMENT SECTION SUMMARY':^80}")
    print(f"{'='*80}\n")
    
    for i, section in enumerate(sections, 1):
        content = section["content"]
        
        text_body = re.sub(r'^#{1,6}\s+.*', '', content, flags=re.MULTILINE).strip()
        
        words = text_body.split()
        word_count = len(words)
        
        sentence_match = re.split(r'(?<=[.!?])\s+', text_body)
        first_sentence = sentence_match[0].replace('\n', ' ').strip() if text_body else "(Empty Section)"
        
        indent = "  " * (section["level"] if section["level"] > 0 else 1)
        header_prefix = f"H{section['level']}: " if section["level"] > 0 else ""
        
        print(f"{indent}Section {i}: {header_prefix}**{section['header']}**")
        print(f"{indent}├── Word Count: {word_count} words")
        print(f"{indent}└── Summary   : \"{first_sentence}\"")
        print()


# Test Verification
if __name__ == "__main__":
    test_document = """
# Lesson 1: Chunking Strategies for RAG

## Overview

Chunking is the foundational step of any Retrieval-Augmented Generation (RAG) pipeline. How you split your documents into smaller pieces directly determines retrieval quality, embedding relevance, and ultimately the answers your system produces. Poor chunking destroys context; thoughtful chunking preserves it.

## Learning Objectives

- Understand why chunking is critical for RAG system performance
- Implement fixed-size, sentence-based, and recursive character text splitting
- Apply semantic chunking and document-aware chunking techniques
- Compare chunking strategies using quantitative metrics
- Choose appropriate chunk sizes and overlap for different use cases

---

## Theory: Why Chunking Matters

When you embed a document for retrieval, the embedding model compresses the entire text into a single vector. If that text is 10,000 tokens long, the vector captures a blurry average of all topics -- making it useless for pinpointing specific information. Chunking breaks documents into focused, semantically coherent pieces so that each embedding represents a clear, retrievable concept.

Key trade-offs:
- **Chunks too large** -- embeddings lose specificity, retrieval returns irrelevant blocks
- **Chunks too small** -- critical context is lost, the LLM receives incomplete information
- **Overlap helps** -- prevents information from being split across chunk boundaries

---

## Theory: Fixed-Size Chunking

The simplest approach: split text every N characters.
""".strip()

    summarize_markdown_sections(test_document)
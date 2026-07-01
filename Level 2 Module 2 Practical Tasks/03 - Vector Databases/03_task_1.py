# Task 1: Build a Document Ingestion Pipeline
# Create a pipeline that:
# Reads all .txt and .md files from a directory
# Chunks each file using RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
# Generates embeddings for each chunk
# Inserts them into a Qdrant collection with metadata (source file, chunk index, file type)
# Handles duplicate IDs by using a hash of the file path + chunk index

import os
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def get_embedding(text: str, dim: int = 384) -> list[float]:
    import random
    random.seed(hash(text))
    return [random.uniform(-1, 1) for _ in range(dim)]

def ingest_directory(directory_path: str, collection_name: str):
    client = QdrantClient(":memory:")
    dim = 384
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    points_to_upsert = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            file_type = "markdown" if filename.endswith(".md") else "text"
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = splitter.split_text(content)
            
            for index, chunk in enumerate(chunks):
                unique_str = f"{file_path}_{index}"
                deterministic_id = int(hashlib.md5(unique_str.encode()).hexdigest(), 16) % (2**63 - 1)
                
                point = PointStruct(
                    id=deterministic_id,
                    vector=get_embedding(chunk, dim=dim),
                    payload={
                        "source_file": file_path,
                        "chunk_index": index,
                        "file_type": file_type,
                        "content": chunk
                    }
                )
                points_to_upsert.append(point)

    if points_to_upsert:
        client.upsert(collection_name=collection_name, points=points_to_upsert)
        print(f"Ingested {len(points_to_upsert)} chunks successfully into '{collection_name}'.")
    else:
        print("No compatible documents discovered or processed.")
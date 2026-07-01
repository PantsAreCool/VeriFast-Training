# Task 3: Production Embedding Pipeline
# Build a production-ready embedding pipeline class that:
# Reads documents from a directory (PDF, TXT, MD files)
# Chunks documents using RecursiveCharacterTextSplitter
# Embeds chunks in configurable batch sizes
# Caches embeddings to avoid re-computation
# Handles rate limiting with exponential backoff
# Logs progress and cost estimates
# Outputs embeddings in a format ready for Qdrant ingestion

import os
import glob
import json
import time
import math
import hashlib
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import tiktoken
from openai import OpenAI, APIConnectionError, RateLimitError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("EmbeddingPipeline")


class ProductionEmbeddingPipeline:
    def __init__(
            self, model_name: str = "openai/text-embedding-3-small", chunk_size: int = 500, chunk_overlap: int = 50, 
            cache_dir: str = ".embedding_cache", max_retries: int = 5, initial_backoff: float = 2.0
            ):

        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_key:
            raise ValueError("Environment variable 'OPENROUTER_API_KEY' is missing.")

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_key)
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=self._count_tokens, 
            separators=["\n\n", "\n", " ", ""]
            )
        
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")


        self.pricing_catalog = {
            "openai/text-embedding-3-small": 0.02,
            "openai/text-embedding-3-large": 0.13,
            "openai/text-embedding-ada-002": 0.10,
        }

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _get_cache_key(self, text: str) -> str:
        payload_string = f"{self.model_name}:{text}"
        return hashlib.md5(payload_string.encode("utf-8")).hexdigest()

    def _read_file_content(self, file_path: Path) -> str:
        """Reads text extracted across supported extensions."""
        ext = file_path.suffix.lower()
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "".join([page.extract_text() or "" for page in reader.pages])
        return ""

    def _embed_with_retry(self, batch_texts: List[str]) -> List[List[float]]:
        """Executes API embedding requests inside a loop(Exponential Backoff)."""
        backoff = self.initial_backoff
        cleaned_batch = [t.replace("\n", " ").strip() for t in batch_texts]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.model_name
                )
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError) as e:
                if attempt == self.max_retries:
                    logger.critical(f"API failed completely after {self.max_retries} attempts.")
                    raise e
                logger.warning(f"Rate limited or Connection dropped. Attempt {attempt}/{self.max_retries}. Retrying in {backoff}s... Error: {e}")
                time.sleep(backoff)
                backoff *= 2  # Exponential progression
            except Exception as e:
                logger.error(f"Unrecoverable structural exception captured during API call: {e}")
                raise e
        return []

    def process_directory(self, source_dir: str, batch_size: int = 32) -> List[PointStruct]:
        """
        Scans directory, extracts texts, splits into chunks, handles contextual metadata, 
        resolves embeddings, logs cost, and outputs Qdrant points.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

        valid_extensions = ["*.txt", "*.md", "*.pdf"]
        file_list = []
        for ext in valid_extensions:
            file_list.extend(glob.glob(os.path.join(source_dir, ext)))

        logger.info(f"Found {len(file_list)} files matching text formats in directory '{source_dir}'.")
        
        all_chunks: List[Dict[str, Any]] = []
        total_tokens_processed = 0

        for file_str in file_list:
            fp = Path(file_str)
            raw_text = self._read_file_content(fp)
            if not raw_text.strip():
                continue
            
            chunks = self.splitter.split_text(raw_text)
            for idx, text_chunk in enumerate(chunks):
                tokens = self._count_tokens(text_chunk)
                total_tokens_processed += tokens
                all_chunks.append({
                    "text": text_chunk,
                    "metadata": {
                        "source_file": fp.name,
                        "file_path": str(fp.absolute()),
                        "chunk_index": idx,
                        "token_count": tokens
                    }
                })

        if not all_chunks:
            logger.info("No text chunks generated to process.")
            return []

        logger.info(f"Generated {len(all_chunks)} unique text chunks. Total token volume: {total_tokens_processed}")
        
        cost_per_million = self.pricing_catalog.get(self.model_name, 0.02)
        estimated_cost = (total_tokens_processed / 1_000_000) * cost_per_million
        logger.info(f"Estimated financial cost for entire run (if completely uncached): ${estimated_cost:.6f} USD")

        qdrant_points: List[PointStruct] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        cache_hits = 0
        for idx, chunk in enumerate(all_chunks):
            cache_key = self._get_cache_key(chunk["text"])
            cache_path = self.cache_dir / f"{cache_key}.json"

            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cache_key))
                    
                    point = PointStruct(
                        id=point_id,
                        vector=cached_data["embedding"],
                        payload={**chunk["metadata"], "text": chunk["text"]}
                    )
                    qdrant_points.append(point)
                    cache_hits += 1
                except Exception:
                    uncached_texts.append(chunk["text"])
                    uncached_indices.append(idx)
            else:
                uncached_texts.append(chunk["text"])
                uncached_indices.append(idx)

        logger.info(f"Cache Performance Lookup: {cache_hits} Hits | {len(uncached_texts)} Misses.")

        if uncached_texts:
            logger.info(f"Processing {len(uncached_texts)} remaining texts via OpenRouter Endpoint using batch sizes of {batch_size}...")
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_slice = uncached_texts[i: i + batch_size]
                global_indices = uncached_indices[i: i + batch_size]
                
                batch_embeddings = self._embed_with_retry(batch_slice)
                
                for local_idx, embedding in enumerate(batch_embeddings):
                    original_chunk_idx = global_indices[local_idx]
                    target_chunk = all_chunks[original_chunk_idx]
                    
                    c_key = self._get_cache_key(target_chunk["text"])
                    c_path = self.cache_dir / f"{c_key}.json"
                    
                    with open(c_path, "w", encoding="utf-8") as f:
                        json.dump({"embedding": embedding, "model": self.model_name}, f)
                    
                    p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c_key))
                    point = PointStruct(
                        id=p_id,
                        vector=embedding,
                        payload={**target_chunk["metadata"], "text": target_chunk["text"]}
                    )
                    qdrant_points.append(point)
                
                logger.info(f"Successfully processed { (i // batch_size) + 1}/{math.ceil(len(uncached_texts)/batch_size)}")

        logger.info(f"Pipeline Completed. Prepared total of {len(qdrant_points)} PointStruct")
        return qdrant_points
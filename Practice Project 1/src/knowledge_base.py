import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class KnowledgeBase:
    """
    Kknowledge base enabling keyword search, relevance scoring, and LLM context retrieval for log analysis.
    """
    def __init__(self):
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 1
        
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 
            'with', 'is', 'was', 'were', 'am', 'are', 'be', 'been', 'of'
        }

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into tokens, lowercases, and removes stopwords"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords]

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Creates and stores a document from log analysis data."""
        doc_id = self.next_id
        self.documents[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "created_at": datetime.now().astimezone().isoformat()
        }
        self.next_id += 1
        return doc_id

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """keyword-based search with relevance score."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored_results = []

        for doc_id, doc in self.documents.items():
            doc_tokens = set(self._tokenize(doc["text"]))
            
            matches = query_tokens.intersection(doc_tokens)
            
            if matches:
                score = len(matches) / math.sqrt(len(query_tokens) * len(doc_tokens))
                
                if doc["metadata"].get("level") in ["ERROR", "CRITICAL", "FATAL"]:
                    score *= 1.2

                scored_results.append({"doc": doc, "score": round(score, 4)})

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:limit]

    def build_llm_context(self, query: str, limit: int = 2) -> str:
        """Retrieves top matches and compiles them into a clean text block ready for direct injection into an LLM prompt."""
        results = self.search(query, limit=limit)
        
        if not results:
            return "No relevant historical log documents or known patterns found in the knowledge base."
            
        context_blocks = []
        for i, res in enumerate(results, 1):
            doc = res["doc"]
            meta = doc["metadata"]
            
            block = (
                f"KB Document #{i} [Relevance Score: {res['score']}]\n"
                f"Content: {doc['text']}\n"
                f"Metadata: Timestamp={meta.get('timestamp', 'N/A')} | "
                f"Level={meta.get('level', 'INFO')} | Source={meta.get('source', 'unknown')}\n"
            )
            context_blocks.append(block)
            
        return "\n".join(context_blocks)
from config import Config
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

class SearchService:
    def __init__(self, config: Config):
        self.cfg = config
        if getattr(self.cfg, 'use_azure_search', False):
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            endpoint = self.cfg.azure_search_endpoint
            key = self.cfg.azure_search_api_key
            index_name = self.cfg.azure_search_index
            self.client = SearchClient(endpoint, index_name, AzureKeyCredential(key))
        else:
            self.embed_model = SentenceTransformer(self.cfg.local_embedding_model)
            self.collection = chroma_client.get_or_create_collection("rag_docs")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for documents matching the query."""
        if not query or not query.strip():
            return []
            
        try:
            if getattr(self.cfg, 'use_azure_search', False):
                results = self.client.search(query, top=top_k)
                contents = []
                for r in results:
                    contents.append(r.get("content", ""))
                return contents
            else:
                q_emb = self.embed_model.encode([query])
                results = self.collection.query(
                    query_embeddings=q_emb.tolist(),
                    n_results=top_k
                )
                
                if not results or not results.get("documents") or not results["documents"]:
                    return []
                    
                return results["documents"][0]
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search and return results with metadata."""
        try:
            q_emb = self.embed_model.encode([query])
            results = self.collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=top_k
            )
            
            if not results or not results.get("documents"):
                return []
            
            # Combine documents with metadata
            search_results = []
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[{}] * len(documents)])[0]
            
            for doc, meta in zip(documents, metadatas):
                search_results.append({
                    "content": doc,
                    "metadata": meta
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Error during search with metadata: {e}")
            return []
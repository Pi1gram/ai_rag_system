from config import Config
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./chroma_db")

class SearchService:
    def __init__(self, config: Config):
        self.cfg = config
        if self.cfg.use_azure_search:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            endpoint = self.cfg.azure_search_endpoint
            key = self.cfg.azure_search_api_key
            index_name = self.cfg.azure_search_index
            self.client = SearchClient(endpoint, index_name, AzureKeyCredential(key))
        else:
            self.embed_model = SentenceTransformer(self.cfg.local_embedding_model)
            self.collection = chroma_client.get_or_create_collection("rag_docs")

    def index_documents(self, docs: List[dict]):
        if self.cfg.use_azure_search:
            actions = []
            for d in docs:
                actions.append({"@search.action": "mergeOrUpload", "id": d["id"], "content": d["content"]})
            result = self.client.upload_documents(documents=actions)
            print(f"Uploaded {len(result)} documents to Azure Search.")
        else:
            raise RuntimeError("Local indexing should be done via ingestor; this method not used locally.")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if self.cfg.use_azure_search:
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
            return results["documents"][0] if results["documents"] else []
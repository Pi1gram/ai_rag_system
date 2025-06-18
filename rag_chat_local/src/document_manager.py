import chromadb
from config import Config
from typing import List, Dict

class DocumentManager:
    def __init__(self, config: Config):
        self.cfg = config
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
    def list_documents(self) -> List[Dict]:
        """List all indexed documents"""
        try:
            collection = self.chroma_client.get_or_create_collection("rag_docs")
            results = collection.get()
            documents = {}
            for metadata in results['metadatas']:
                source = metadata.get('source', 'Unknown')
                if source not in documents:
                    documents[source] = 0
                documents[source] += 1
            return [{"name": k, "chunks": v} for k, v in documents.items()]
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        try:
            collection = self.chroma_client.get_or_create_collection("rag_docs")
            count = collection.count()
            return {
                "total_chunks": count,
                "total_documents": len(self.list_documents()),
                "embedding_model": self.cfg.local_embedding_model
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "total_documents": 0, "embedding_model": "unknown"}
from pathlib import Path
from search_service import SearchService
from config import Config
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

chroma_client = chromadb.PersistentClient(path="./chroma_db")


# Utility: read text files
def load_texts(folder):
    texts = []
    paths = []
    
    # Text files
    for ext in (".txt", ".md"):
        for file in Path(folder).glob(f"*{ext}"):
            texts.append(file.read_text(encoding="utf-8", errors="ignore"))
            paths.append(str(file))
    
    # PDFs
    for file in Path(folder).glob("*.pdf"):
        try:
            reader = PdfReader(str(file))
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
            texts.append(pdf_text)
            paths.append(str(file))
        except Exception as e:
            print(f"Could not read {file}: {e}")
    
    # DOCX files (optional - requires python-docx)
    try:
        import docx
        for file in Path(folder).glob("*.docx"):
            try:
                doc = docx.Document(str(file))
                docx_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                texts.append(docx_text)
                paths.append(str(file))
            except Exception as e:
                print(f"Could not read {file}: {e}")
    except ImportError:
        print("python-docx not installed - skipping .docx files")
    
    return texts, paths


# Chunking by words
def chunk_text(text, size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks


class Ingestor:
    def __init__(self, config: Config):
        self.cfg = config
        self.embed_model = SentenceTransformer(self.cfg.local_embedding_model)
        self.search_service = SearchService(self.cfg)

    def ingest(self):
        texts, paths = load_texts(self.cfg.data_folder)
        print(f"Loaded files: {paths}")
        all_docs = []
        for text, path in zip(texts, paths):
            chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{Path(path).stem}_{idx}"
                all_docs.append({"id": doc_id, "content": chunk, "source": str(path)})
        print(f"Total chunks: {len(all_docs)}")
        
        if not all_docs:
            print("No documents found to ingest!")
            return
            
        texts = [d["content"] for d in all_docs]
        embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        # Use Chroma only
        collection = chroma_client.get_or_create_collection(name="rag_docs")
        for doc, emb in zip(all_docs, embeddings):
            collection.add(
                ids=[doc["id"]],
                embeddings=[emb.tolist()],
                documents=[doc["content"]],
                metadatas=[{"source": doc["source"]}]
            )
        print("Chroma DB collection updated.")

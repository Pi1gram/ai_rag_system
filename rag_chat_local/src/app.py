import argparse
from config import Config
from ingest import Ingestor
from search_service import SearchService
from llm_service import LLMService
import gradio as gr

cfg = Config()

def ingest_command():
    ingestor = Ingestor(cfg)
    ingestor.ingest()

def generate_answer(llm_service: LLMService, search_service: SearchService, query: str) -> str:
    try:
        chunks = search_service.search(query, top_k=5)
    except Exception as e:
        return f"Error during retrieval: {e}"

    prompt = "You are an assistant. Use the following excerpts to answer the question.\n\n"
    for idx, chunk in enumerate(chunks, 1):
        prompt += f"Excerpt {idx}:\n{chunk}\n\n"
    prompt += f"Question: {query}\nAnswer:"

    try:
        ans = llm_service.generate(prompt)
    except Exception as e:
        return f"Error during generation: {e}"
    
    return ans

def get_document_count():
    """Helper function to get document count"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("rag_docs")
        return str(collection.count())
    except:
        return "0"

def get_chunk_count():
    """Helper function to get chunk count"""
    return get_document_count()  # Same as document count for now

def chat_interface():
    llm = LLMService(cfg)
    search = SearchService(cfg)

    def chat_fn(message, history):
        answer = generate_answer(llm, search, message)
        return answer

    # Modern chat interface with streaming
    with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chat Local") as demo:
        gr.Markdown("# 🤖 RAG Chat Local\n**Ask questions about your documents**")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    type="messages",
                    examples=[
                        "What are the main topics in my documents?",
                        "Summarize the key points",
                        "Tell me about Percy the pig"
                    ],
                    title="Chat with your documents",
                    description="I can answer questions based on your uploaded documents."
                )
            
            with gr.Column(scale=1):
                with gr.Accordion("📊 Document Stats", open=False):
                    doc_count = gr.Textbox(label="Documents", value=get_document_count(), interactive=False)
                    chunk_count = gr.Textbox(label="Chunks", value=get_chunk_count(), interactive=False)
                
                with gr.Accordion("⚙️ Settings", open=False):
                    temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                    top_k = gr.Slider(1, 10, value=5, label="Top-K Results")
        
        demo.launch(share=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', action='store_true', help='Run ingestion to build index')
    args = parser.parse_args()
    
    if args.ingest:
        ingest_command()
    else:
        chat_interface()
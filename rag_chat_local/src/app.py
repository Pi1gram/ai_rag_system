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

def chat_interface():
    llm = LLMService(cfg)
    search = SearchService(cfg)

    def chat_fn(user_input, history):
        answer = generate_answer(llm, search, user_input)
        history = history + [(user_input, answer)]
        return history, history

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        user = gr.Textbox(placeholder="Ask about your lectures...")
        user.submit(chat_fn, [user, chatbot], [chatbot, chatbot])
        demo.launch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', action='store_true', help='Run ingestion to build index')
    args = parser.parse_args()
    
    if args.ingest:
        ingest_command()
    else:
        chat_interface()
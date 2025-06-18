class Config:
    def __init__(self, path="config.yaml"):
        import yaml
        from pathlib import Path

        cfg = yaml.safe_load(Path(path).read_text())

        # Azure toggles
        self.use_azure_search = cfg.get("use_azure_search", False)
        self.use_azure_openai = cfg.get("use_azure_openai", False)
        self.use_gemini = cfg.get("use_gemini", False)

        # Azure Search
        self.azure_search_endpoint = cfg.get("azure_search_endpoint", "")
        self.azure_search_api_key = cfg.get("azure_search_api_key", "")
        self.azure_search_index = cfg.get("azure_search_index", "lectures")

        # Azure OpenAI
        self.azure_openai_endpoint = cfg.get("azure_openai_endpoint", "")
        self.azure_openai_key = cfg.get("azure_openai_key", "")
        self.azure_openai_deployment = cfg.get("azure_openai_deployment", "")

        # Gemini
        self.gemini_api_key = cfg.get("gemini_api_key", "")

        # Local
        self.local_embedding_model = cfg.get(
            "local_embedding_model", "all-MiniLM-L6-v2"
        )
        self.chunk_size = cfg.get("chunk_size", 400)
        self.chunk_overlap = cfg.get("chunk_overlap", 50)
        self.local_llm_model_path = cfg.get(
            "local_llm_model_path", "./models/llama-2-7b-chat.Q4_K_M.gguf"
        )

        # Ingestion
        self.data_folder = cfg.get("data_folder", "./data")

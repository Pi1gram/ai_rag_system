from config import Config

class LLMService:
    def __init__(self, config: Config):
        self.cfg = config
        if getattr(self.cfg, "use_gemini", False):
            import google.generativeai as genai
            genai.configure(api_key=self.cfg.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        elif self.cfg.use_azure_openai:
            from azure.ai.openai import OpenAIClient
            from azure.core.credentials import AzureKeyCredential
            self.client = OpenAIClient(self.cfg.azure_openai_endpoint, AzureKeyCredential(self.cfg.azure_openai_key))
            self.deployment = self.cfg.azure_openai_deployment
        else:
            # Only fallback to Llama if not Gemini or Azure
            from llama_cpp import Llama
            if not self.cfg.local_llm_model_path:
                raise ValueError("Local LLM model path not set in config.")
            self.llm = Llama(model_path=self.cfg.local_llm_model_path)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if getattr(self.cfg, "use_gemini", False):
            response = self.model.generate_content(prompt)
            return response.text
        elif self.cfg.use_azure_openai:
            resp = self.client.get_chat_completions(
                deployment_id=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        else:
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                stop=["</s>"]
            )
            return output['choices'][0]['text']
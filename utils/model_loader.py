import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from configs.logger import GLOBAL_LOGGER as log
from configs.exceptions import DocumentPortalException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeyliveclass")

        # Load from JSON secret (ECS etc.)
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS", error=str(e))

        # Fallback to env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from env")

        # Validate
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded")

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embeddings and LLM based on your Astra RAG config.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()

        log.info("Config loaded", keys=list(self.config.keys()))

    # -------------------------
    # Embeddings
    # -------------------------
    def load_embeddings(self):
        """
        Load HuggingFace embeddings (as per config).
        """
        try:
            model_name = self.config["embedding_model"].get("model_name", "default")

            if model_name == "default":
                log.info("Loading default HuggingFace embeddings")
                return HuggingFaceEmbeddings()

            log.info("Loading HuggingFace embeddings", model=model_name)
            return HuggingFaceEmbeddings(model_name=model_name)

        except Exception as e:
            log.error("Failed to load embeddings", error=str(e))
            raise DocumentPortalException("Embedding load failed", sys)

    # -------------------------
    # LLM
    # -------------------------
    def load_llm(self):
        """
        Load Groq LLM (flat config).
        """
        try:
            llm_config = self.config["llm"]

            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "groq":
                return ChatGroq(
                    model_name=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM load failed", sys)


# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    loader = ModelLoader()

    # Test embeddings
    embeddings = loader.load_embeddings()
    print("Embeddings Loaded:", embeddings)

    result = embeddings.embed_query("Hello RAG")
    print("Embedding vector size:", len(result))

    # Test LLM
    llm = loader.load_llm()
    print("LLM Loaded:", llm)

    result = llm.invoke("Explain RAG in one line")
    print("LLM Response:", result.content)

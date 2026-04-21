from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-05-01-preview"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_index_name: str = "enterprise-kb-index"

    # Azure AI Content Safety
    azure_content_safety_endpoint: str

    # Azure Key Vault
    azure_keyvault_url: str = ""

    # Chunking — per source type
    docs_chunk_size: int = 512
    docs_chunk_overlap: int = 64
    tickets_chunk_size: int = 256
    changelog_chunk_size: int = 9999

    # Retrieval
    top_k_results: int = 5
    rerank_top_n: int = 20

    # Deduplication
    dedup_similarity_threshold: float = 0.97

    # Generation
    max_tokens_response: int = 800

    # Safety
    content_safety_threshold: int = 2

    # Logging
    log_level: str = "INFO"


settings = Settings()
"""
Root conftest.py — stubs azure.* and sets env vars before any test import.
"""

import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
os.environ.setdefault("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_INDEX_NAME", "test-index")
os.environ.setdefault("AZURE_CONTENT_SAFETY_ENDPOINT", "https://test.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_KEYVAULT_URL", "https://test-vault.vault.azure.net/")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def _stub():
    mods = [
        "azure", "azure.identity", "azure.keyvault", "azure.keyvault.secrets",
        "azure.search", "azure.search.documents", "azure.search.documents.indexes",
        "azure.search.documents.indexes.models", "azure.search.documents.models",
        "azure.ai", "azure.ai.contentsafety", "azure.ai.contentsafety.models",
        "azure.ai.evaluation", "azure.storage", "azure.storage.blob",
        "azure.core", "azure.core.exceptions",
        "openai", "tiktoken", "structlog", "tenacity",
        "pandas",
    ]
    for mod in mods:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


_stub()
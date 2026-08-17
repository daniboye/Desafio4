"""
Modulo: config.py

Centraliza a leitura de variaveis de ambiente e chaves de API.
python-dotenv le o arquivo .env (uso local). Em producao no Streamlit
Community Cloud, a chave vem da secao Secrets, exposta automaticamente
como variavel de ambiente -- por isso o mesmo codigo funciona nos dois
cenarios sem alteracao.

O provedor de LLM (Groq ou Gemini) e escolhido pela variavel de ambiente
LLM_PROVIDER; o padrao e "groq" quando essa variavel nao e definida.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def obter_provedor_llm() -> str:
    """Devolve o provedor de LLM configurado ("groq" ou "gemini"), com "groq" como padrao."""
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def obter_chave_api_groq():
    """Devolve a chave de API da Groq configurada no ambiente, ou None."""
    return os.getenv("GROQ_API_KEY")


def obter_chave_api_google():
    """Devolve a chave de API do Gemini configurada no ambiente, ou None."""
    return os.getenv("GOOGLE_API_KEY")

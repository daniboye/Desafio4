"""
Modulo: config.py

Centraliza a leitura de variaveis de ambiente e chaves de API.
python-dotenv le o arquivo .env (uso local). Em producao no Streamlit
Community Cloud, a chave vem da secao Secrets, exposta automaticamente
como variavel de ambiente -- por isso o mesmo codigo funciona nos dois
cenarios sem alteracao.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def obter_chave_api_google():
    """Devolve a chave de API do Gemini configurada no ambiente, ou None."""
    return os.getenv("GOOGLE_API_KEY")

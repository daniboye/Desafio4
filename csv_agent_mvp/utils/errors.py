"""
Modulo: utils/errors.py

Reuniamos aqui um unico lugar para tratar erros "de fronteira": aqueles
que acontecem na interacao com o usuario (upload invalido, pergunta
vazia, chave de API ausente) e que devem virar mensagens amigaveis na
tela, em vez de uma tela de erro tecnica assustadora.
"""


def mensagem_amigavel_erro_upload(erro: Exception) -> str:
    """Traduz um erro de carga de arquivo em uma mensagem clara para o usuario."""
    return (
        f"Nao foi possivel processar o arquivo enviado: {erro} "
        f"Verifique se voce enviou um arquivo .zip contendo pelo menos "
        f"um arquivo .csv e tente novamente."
    )


def mensagem_amigavel_erro_agente(erro: Exception) -> str:
    """Traduz um erro do agente/LLM em uma mensagem clara para o usuario."""
    texto_erro = str(erro).lower()

    if "api key" in texto_erro or "credential" in texto_erro or "chave" in texto_erro:
        return (
            "Nao foi possivel conectar ao modelo de IA: a chave de API nao "
            "foi configurada corretamente. Verifique a configuracao de "
            "GOOGLE_API_KEY."
        )

    if "quota" in texto_erro or "rate limit" in texto_erro or "429" in texto_erro:
        return (
            "O limite de uso gratuito da IA foi atingido no momento. "
            "Aguarde um minuto e tente novamente."
        )

    return (
        f"Ocorreu um erro ao processar sua pergunta: {erro} "
        f"Tente reformular a pergunta ou verifique se ela se refere aos "
        f"dados carregados."
    )

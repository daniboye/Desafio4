"""
Modulo: agent/build_agent.py

Este e o modulo que "monta" o agente de verdade: junta tres pecas --

  1) o MODELO DE LINGUAGEM (o Gemini, via langchain-google-genai);
  2) as FERRAMENTAS (as funcoes em agent/tools/, que fazem os calculos);
  3) o PROMPT DE SISTEMA (as instrucoes de comportamento, em prompts.py);

-- e devolve um "agente" pronto para receber uma pergunta em portugues e
devolver uma resposta, decidindo sozinho quais ferramentas chamar pelo
caminho.

NOTA DE VERSAO: este codigo usa a API "create_agent", introduzida no
LangChain 1.0 (lancado em 2025). Versoes anteriores do LangChain (0.x)
usavam "create_tool_calling_agent" + "AgentExecutor", que foram
descontinuadas nessa nova versao principal. Usamos aqui a API mais nova
porque e a que o Streamlit Community Cloud instala por padrao (ele sempre
busca a versao mais recente compativel disponivel).

Conceito-chave "tool calling": o modelo Gemini foi treinado para,
recebendo a descricao de uma funcao Python (nome, parametros, docstring),
DECIDIR quando faz sentido chama-la e com quais valores. O LangChain e
quem faz a ponte entre "o modelo pediu para chamar tal funcao com tais
parametros" e "chamar de fato essa funcao Python e devolver o resultado
ao modelo".
"""

import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.prompts import PROMPT_SISTEMA
from agent.tools.query_tools import consultar_dataframe, resumo_estatistico
from agent.tools.chart_tools import gerar_grafico_a_partir_de_tabela
from langchain_core.tools import tool


# Envolvemos a funcao de graficos com @tool aqui (em vez de no proprio
# chart_tools.py) para manter aquele modulo mais simples de testar sem
# depender do LangChain instalado -- uma escolha de organizacao, nao uma
# regra obrigatoria.
@tool
def gerar_grafico(tabela_markdown: str, tipo_grafico: str = "barra", titulo: str = "") -> str:
    """
    Gera um grafico (barra, linha ou pizza) a partir do resultado de uma
    consulta anterior feita com a ferramenta consultar_dataframe. Use
    esta ferramenta sempre que a pergunta pedir uma comparacao entre
    categorias, uma evolucao no tempo, ou um ranking visual.

    Args:
        tabela_markdown: o texto da tabela retornado por consultar_dataframe.
        tipo_grafico: "barra", "linha" ou "pizza".
        titulo: um titulo descritivo para o grafico.
    """
    return gerar_grafico_a_partir_de_tabela(tabela_markdown, tipo_grafico, titulo)


class AgenteCsv:
    """
    Pequeno "envelope" ao redor do agente do LangChain, para que o resto
    do nosso codigo (app.py) tenha um jeito simples e estavel de chamar
    o agente -- passando uma pergunta em texto e recebendo uma resposta
    em texto -- sem se preocupar com o formato interno de mensagens que
    o LangChain usa por baixo dos panos.
    """

    def __init__(self, agente_langchain):
        self._agente = agente_langchain

    def invoke(self, entrada: dict) -> dict:
        """
        Recebe {"input": "pergunta do usuario"} (mesmo formato usado antes,
        para nao precisarmos alterar o app.py), e devolve {"output": "resposta"}.

        Por baixo dos panos, a nova API do LangChain (create_agent) espera
        e devolve uma LISTA DE MENSAGENS (formato "messages"), entao
        fazemos essa conversao aqui dentro.
        """
        pergunta = entrada["input"]
        resultado = self._agente.invoke({"messages": [{"role": "user", "content": pergunta}]})
        ultima_mensagem = resultado["messages"][-1]
        return {"output": ultima_mensagem.content}


def montar_agente(schema_context: str, chave_api_google: str | None = None) -> AgenteCsv:
    """
    Constroi e devolve um agente pronto para uso (envolto em AgenteCsv).

    Parametros:
      schema_context: o texto gerado por data/schema.py, descrevendo as
        tabelas e colunas disponiveis nesta sessao.
      chave_api_google: a chave de API do Gemini. Se nao for informada,
        tenta ler da variavel de ambiente GOOGLE_API_KEY (vinda do .env
        localmente, ou dos "Secrets" do Streamlit Cloud em producao).

    Devolve: um AgenteCsv, que tem um metodo .invoke({"input": pergunta})
    usado pela interface para obter a resposta.
    """
    chave = chave_api_google or os.getenv("GOOGLE_API_KEY")
    if not chave:
        raise ValueError(
            "Chave de API do Gemini nao encontrada. Configure a variavel "
            "de ambiente GOOGLE_API_KEY (arquivo .env local, ou 'Secrets' "
            "no Streamlit Cloud)."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=chave,
        temperature=0,  # temperatura baixa = respostas mais consistentes e
                        # menos "criativas", ideal para tarefas analiticas.
    )

    ferramentas = [consultar_dataframe, resumo_estatistico, gerar_grafico]

    prompt_sistema_preenchido = PROMPT_SISTEMA.format(schema_context=schema_context)

    agente_langchain = create_agent(
        model=llm,
        tools=ferramentas,
        system_prompt=prompt_sistema_preenchido,
    )

    return AgenteCsv(agente_langchain)

"""
Modulo: agent/build_agent.py

Este e o modulo que "monta" o agente de verdade: junta tres pecas --

  1) o MODELO DE LINGUAGEM (Groq ou Gemini, configuravel);
  2) as FERRAMENTAS (as funcoes em agent/tools/, que fazem os calculos);
  3) o PROMPT DE SISTEMA (as instrucoes de comportamento, em prompts.py);

-- e devolve um "agente" pronto para receber uma pergunta em portugues e
devolver uma resposta, decidindo sozinho quais ferramentas chamar pelo
caminho.

NOTA DE VERSAO 1: este codigo usa a API "create_agent", introduzida no
LangChain 1.0 (lancado em 2025). Versoes anteriores do LangChain (0.x)
usavam "create_tool_calling_agent" + "AgentExecutor", que foram
descontinuadas nessa nova versao principal.

NOTA DE VERSAO 2: o provedor de LLM agora e configuravel via a variavel
de ambiente LLM_PROVIDER ("groq" ou "gemini"). O motivo da mudanca para
Groq como padrao: a cota gratuita do Gemini se mostrou instavel (o nome
de modelos especificos e periodicamente descontinuado, e o limite diario
gratuito e baixo o suficiente para esgotar durante testes normais). O
Groq oferece uma cota gratuita mais generosa e exclusiva por conta,
alem de respostas mais rapidas (hardware LPU dedicado). O Gemini
permanece disponivel como alternativa configuravel, sem precisar alterar
codigo -- apenas trocar a variavel de ambiente e a chave correspondente.

NOTA DE VERSAO 3: nas versoes mais novas do langchain-google-genai, o
campo ".content" da resposta do modelo pode vir como uma LISTA DE BLOCOS
estruturados (cada bloco com um "type", geralmente "text", mais um texto
e metadados internos do Google como "extras.signature"), em vez de vir
como um texto simples direto. A funcao _extrair_texto_da_resposta(),
abaixo, trata os dois formatos, garantindo compatibilidade tanto com
Groq quanto com Gemini.

Conceito-chave "tool calling": o modelo foi treinado para, recebendo a
descricao de uma funcao Python (nome, parametros, docstring), DECIDIR
quando faz sentido chama-la e com quais valores. O LangChain e quem faz
a ponte entre "o modelo pediu para chamar tal funcao com tais parametros"
e "chamar de fato essa funcao Python e devolver o resultado ao modelo".
"""

import os
from langchain.agents import create_agent
from langchain_core.tools import tool

from agent.prompts import PROMPT_SISTEMA
from agent.tools.query_tools import consultar_dataframe, resumo_estatistico
from agent.tools.chart_tools import gerar_grafico_a_partir_de_tabela


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


def _extrair_texto_da_resposta(conteudo) -> str:
    """
    Converte o campo ".content" de uma mensagem do modelo em uma string
    de texto simples, pronta para ser exibida ao usuario.

    Trata os dois formatos possiveis:
      1) uma string simples (formato mais comum, inclusive no Groq):
         devolvida direto.
      2) uma lista de blocos estruturados (formato usado por versoes mais
         novas do conector do Gemini): percorremos a lista e juntamos
         apenas o texto de cada bloco do tipo "text", descartando
         metadados internos como "extras" (sem valor para o usuario final).
    """
    if isinstance(conteudo, str):
        return conteudo

    if isinstance(conteudo, list):
        pedacos_de_texto = []
        for bloco in conteudo:
            if isinstance(bloco, dict) and bloco.get("type") == "text":
                pedacos_de_texto.append(bloco.get("text", ""))
            elif isinstance(bloco, str):
                pedacos_de_texto.append(bloco)
        return "\n".join(pedacos_de_texto).strip()

    # Formato inesperado: convertemos para texto simples como ultimo recurso.
    return str(conteudo)


class AgenteCsv:
    """
    Pequeno "envelope" ao redor do agente do LangChain, para que o resto
    do nosso codigo (app.py) tenha um jeito simples e estavel de chamar
    o agente -- passando uma pergunta em texto e recebendo uma resposta
    em texto -- sem se preocupar com o formato interno de mensagens que
    o LangChain usa por baixo dos panos, nem com qual provedor de LLM
    esta sendo usado (Groq ou Gemini).
    """

    def __init__(self, agente_langchain):
        self._agente = agente_langchain

    def invoke(self, entrada: dict) -> dict:
        """
        Recebe {"input": "pergunta do usuario"} e devolve {"output": "resposta"}.

        Por baixo dos panos, a API create_agent do LangChain 1.0 espera e
        devolve uma LISTA DE MENSAGENS (formato "messages"), entao fazemos
        essa conversao aqui dentro.
        """
        pergunta = entrada["input"]
        resultado = self._agente.invoke({"messages": [{"role": "user", "content": pergunta}]})
        ultima_mensagem = resultado["messages"][-1]
        texto_resposta = _extrair_texto_da_resposta(ultima_mensagem.content)
        return {"output": texto_resposta}


def _criar_llm(provedor: str):
    """
    Cria a instancia do modelo de linguagem de acordo com o provedor
    escolhido. Isolar essa decisao em uma funcao separada facilita trocar
    de provedor no futuro sem mexer no restante da montagem do agente.
    """
    if provedor == "groq":
        from langchain_groq import ChatGroq

        chave = os.getenv("GROQ_API_KEY")
        if not chave:
            raise ValueError(
                "Chave de API da Groq nao encontrada. Configure a variavel "
                "de ambiente GROQ_API_KEY (arquivo .env local, ou 'Secrets' "
                "no Streamlit Cloud). Obtenha uma chave gratuita em "
                "https://console.groq.com/keys"
            )
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=chave,
            temperature=0,  # temperatura baixa = respostas mais consistentes,
                            # ideal para tarefas analiticas.
        )

    if provedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        chave = os.getenv("GOOGLE_API_KEY")
        if not chave:
            raise ValueError(
                "Chave de API do Gemini nao encontrada. Configure a variavel "
                "de ambiente GOOGLE_API_KEY (arquivo .env local, ou 'Secrets' "
                "no Streamlit Cloud)."
            )
        return ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=chave,
            temperature=0,
        )

    raise ValueError(
        f"Provedor de LLM '{provedor}' nao reconhecido. Use 'groq' ou 'gemini' "
        f"na variavel de ambiente LLM_PROVIDER."
    )


def montar_agente(schema_context: str, chave_api: str | None = None) -> AgenteCsv:
    """
    Constroi e devolve um agente pronto para uso (envolto em AgenteCsv).

    Parametros:
      schema_context: o texto gerado por data/schema.py, descrevendo as
        tabelas e colunas disponiveis nesta sessao.
      chave_api: opcional; permite passar a chave diretamente em vez de
        depender apenas da variavel de ambiente (usado principalmente em
        testes). Quando informado, e usado independentemente do provedor
        escolhido, sobrescrevendo a leitura de ambiente para essa chamada.

    O provedor (Groq ou Gemini) e escolhido pela variavel de ambiente
    LLM_PROVIDER. Se nao for definida, o padrao e "groq".

    Devolve: um AgenteCsv, que tem um metodo .invoke({"input": pergunta})
    usado pela interface para obter a resposta.
    """
    provedor = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if chave_api:
        variavel = "GROQ_API_KEY" if provedor == "groq" else "GOOGLE_API_KEY"
        os.environ[variavel] = chave_api

    llm = _criar_llm(provedor)

    ferramentas = [consultar_dataframe, resumo_estatistico, gerar_grafico]

    prompt_sistema_preenchido = PROMPT_SISTEMA.format(schema_context=schema_context)

    agente_langchain = create_agent(
        model=llm,
        tools=ferramentas,
        system_prompt=prompt_sistema_preenchido,
    )

    return AgenteCsv(agente_langchain)

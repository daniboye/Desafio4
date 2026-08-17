"""
app.py

Este e o arquivo PRINCIPAL da aplicacao: e ele que o Streamlit executa
quando a aplicacao e iniciada. Ele junta tudo que construimos ate agora:

  - data/loader.py      -> le o .zip enviado pelo usuario;
  - data/schema.py       -> descreve as tabelas para a IA;
  - agent/build_agent.py -> monta o agente (Gemini + ferramentas + prompt);
  - agent/tools/*        -> as ferramentas que o agente usa.

O arquivo esta organizado em duas partes visiveis, que correspondem as
duas interfaces exigidas pelo desafio:

  INTERFACE A (Carga dos dados): aparece enquanto nenhum arquivo valido
  foi carregado ainda.

  INTERFACE B (Consulta): aparece assim que a carga e concluida com
  sucesso, permitindo perguntas em linguagem natural.

Um conceito importante do Streamlit usado aqui: "st.session_state". Um
app Streamlit reexecuta esse arquivo inteiro do zero a cada interacao do
usuario (por exemplo, a cada pergunta enviada). Para nao perder os dados
carregados ou o historico da conversa a cada reexecucao, guardamos essas
informacoes no "st.session_state", que e como uma memoria que sobrevive
entre as reexecucoes, enquanto durar a sessao do navegador.
"""

import streamlit as st

from data.loader import carregar_zip, ErroDeCarga
from data.schema import montar_schema_context
from agent.build_agent import montar_agente
from agent.tools.query_tools import definir_tabelas_ativas
from agent.tools import chart_tools
from utils.errors import mensagem_amigavel_erro_upload, mensagem_amigavel_erro_agente


st.set_page_config(page_title="Consulta Inteligente de CSVs", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Inicializacao do "session_state": na primeira vez que o app roda, criamos
# as variaveis que vamos usar ao longo da sessao do usuario.
# ---------------------------------------------------------------------------
if "dados_carregados" not in st.session_state:
    st.session_state.dados_carregados = False
if "tabelas" not in st.session_state:
    st.session_state.tabelas = {}
if "schema_context" not in st.session_state:
    st.session_state.schema_context = ""
if "agente" not in st.session_state:
    st.session_state.agente = None
if "historico_mensagens" not in st.session_state:
    st.session_state.historico_mensagens = []  # lista de (autor, texto)


st.title("📊 Interface Inteligente para Consulta de Arquivos CSV")
st.caption(
    "Envie um arquivo .zip com seus CSVs e faça perguntas em linguagem "
    "natural sobre os dados."
)


# ---------------------------------------------------------------------------
# INTERFACE A: Carga dos dados
# So aparece (ou permanece disponivel) enquanto o usuario ainda nao carregou
# dados, ou quer carregar um novo conjunto de dados.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Carregar dados")
    arquivo_zip = st.file_uploader(
        "Envie um arquivo .zip contendo um ou mais arquivos .csv "
        "(e, opcionalmente, um dicionário de dados)",
        type=["zip"],
    )

    if arquivo_zip is not None:
        if st.button("Processar arquivo", type="primary"):
            with st.spinner("Processando arquivo..."):
                try:
                    tabelas, dicionario_dados = carregar_zip(arquivo_zip)
                    schema_context = montar_schema_context(tabelas, dicionario_dados)

                    # Avisamos as ferramentas quais tabelas estao ativas agora.
                    definir_tabelas_ativas(tabelas)

                    st.session_state.tabelas = tabelas
                    st.session_state.schema_context = schema_context
                    st.session_state.dados_carregados = True
                    st.session_state.agente = None  # forca recriacao do agente
                    st.session_state.historico_mensagens = []

                    st.success(
                        f"{len(tabelas)} tabela(s) carregada(s) com sucesso!"
                    )
                except ErroDeCarga as erro:
                    st.error(mensagem_amigavel_erro_upload(erro))
                except Exception as erro:
                    st.error(mensagem_amigavel_erro_upload(erro))

    if st.session_state.dados_carregados:
        st.divider()
        st.subheader("Tabelas carregadas")
        for nome_tabela, df in st.session_state.tabelas.items():
            st.write(f"**{nome_tabela}**: {df.shape[0]} linhas, {df.shape[1]} colunas")


# ---------------------------------------------------------------------------
# INTERFACE B: Consulta em linguagem natural
# So e liberada depois que a carga (Interface A) e concluida com sucesso.
# ---------------------------------------------------------------------------
if not st.session_state.dados_carregados:
    st.info(
        "⬅️ Para começar, envie um arquivo .zip com seus arquivos CSV na "
        "barra lateral à esquerda."
    )
else:
    # Cria o agente apenas uma vez por conjunto de dados carregado (nao a
    # cada pergunta), para evitar reconstrui-lo sem necessidade.
    if st.session_state.agente is None:
        try:
            st.session_state.agente = montar_agente(st.session_state.schema_context)
        except ValueError as erro:
            st.error(str(erro))
            st.stop()

    # Exibe o historico de mensagens ja trocadas nesta sessao.
    for autor, conteudo in st.session_state.historico_mensagens:
        with st.chat_message(autor):
            st.markdown(conteudo)

    pergunta_usuario = st.chat_input(
        "Faça uma pergunta sobre os dados carregados..."
    )

    if pergunta_usuario:
        # Bloqueia perguntas vazias/so espacos antes de acionar o agente.
        if not pergunta_usuario.strip():
            st.warning("Digite uma pergunta antes de enviar.")
        else:
            st.session_state.historico_mensagens.append(("user", pergunta_usuario))
            with st.chat_message("user"):
                st.markdown(pergunta_usuario)

            with st.chat_message("assistant"):
                with st.spinner("Consultando os dados..."):
                    # Zera o grafico anterior antes de perguntar de novo, para
                    # nao reexibir por engano um grafico de uma pergunta antiga
                    # caso esta pergunta nao gere um novo grafico.
                    chart_tools.ULTIMO_GRAFICO_GERADO = None

                    try:
                        resultado = st.session_state.agente.invoke(
                            {"input": pergunta_usuario}
                        )
                        resposta_texto = resultado["output"]
                    except Exception as erro:
                        resposta_texto = mensagem_amigavel_erro_agente(erro)

                    st.markdown(resposta_texto)

                    # Se alguma ferramenta gerou um grafico durante esta
                    # pergunta, exibimos ele logo abaixo da resposta em texto.
                    if chart_tools.ULTIMO_GRAFICO_GERADO is not None:
                        st.plotly_chart(
                            chart_tools.ULTIMO_GRAFICO_GERADO,
                            use_container_width=True,
                        )

            st.session_state.historico_mensagens.append(("assistant", resposta_texto))

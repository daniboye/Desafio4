"""
Módulo: agent/tools/chart_tools.py

Esta é a terceira ferramenta do agente: gerar gráficos.

Diferente das ferramentas anteriores (que devolvem texto/tabela), esta
ferramenta devolve um "identificador" de gráfico. O motivo é técnico:
uma ferramenta LangChain deve devolver texto simples (string), mas um
gráfico Plotly é um objeto complexo. Por isso, guardamos o gráfico numa
variável compartilhada (parecido com o que já fizemos com as tabelas em
query_tools.py) e devolvemos só um "recibo" em texto. Quem realmente
desenha o gráfico na tela é a Interface (Streamlit), lendo essa variável
depois que o agente termina de responder.
"""

from typing import Optional
import pandas as pd
import plotly.express as px

# Aqui guardamos o ULTIMO grafico gerado nesta sessao, para a interface
# Streamlit poder exibi-lo apos o agente responder.
ULTIMO_GRAFICO_GERADO = None


def _resultado_texto_para_dataframe(texto_markdown: str) -> pd.DataFrame:
    """
    A ferramenta consultar_dataframe devolve uma tabela em formato de
    texto Markdown (para ser lida pela IA). Esta funcao converte esse
    texto de volta em um DataFrame, para podermos plotar um grafico a
    partir dele.
    """
    linhas = [l for l in texto_markdown.strip().split("\n") if l.strip()]
    if len(linhas) < 2:
        raise ValueError("Nao ha dados suficientes em formato de tabela para gerar um grafico.")

    def _dividir_linha(linha: str) -> list[str]:
        return [celula.strip() for celula in linha.strip("|").split("|")]

    cabecalho = _dividir_linha(linhas[0])
    linhas_de_dados = [_dividir_linha(l) for l in linhas[2:]]

    df = pd.DataFrame(linhas_de_dados, columns=cabecalho)

    for coluna in df.columns:
        convertida = pd.to_numeric(df[coluna].str.replace(",", ""), errors="coerce")
        if convertida.notna().all():
            df[coluna] = convertida

    return df


def gerar_grafico_a_partir_de_tabela(
    tabela_markdown: str,
    tipo_grafico: str = "barra",
    titulo: Optional[str] = None,
) -> str:
    """
    Recebe uma tabela em texto (o resultado de consultar_dataframe) e
    gera um grafico a partir dela, guardando-o para a interface exibir.

    Args:
        tabela_markdown: a tabela em formato de texto Markdown, exatamente
            como devolvida por consultar_dataframe.
        tipo_grafico: "barra", "linha" ou "pizza".
        titulo: titulo a ser exibido no grafico.

    Returns:
        Uma mensagem de texto confirmando que o grafico foi gerado, para
        a IA incluir na resposta ao usuario.
    """
    global ULTIMO_GRAFICO_GERADO

    try:
        df = _resultado_texto_para_dataframe(tabela_markdown)

        if df.shape[1] < 2:
            return "Nao foi possivel gerar o grafico: a tabela precisa de pelo menos duas colunas."

        coluna_categoria = df.columns[0]
        coluna_valor = df.columns[1]

        if tipo_grafico == "linha":
            figura = px.line(df, x=coluna_categoria, y=coluna_valor, title=titulo)
        elif tipo_grafico == "pizza":
            figura = px.pie(df, names=coluna_categoria, values=coluna_valor, title=titulo)
        else:
            figura = px.bar(df, x=coluna_categoria, y=coluna_valor, title=titulo)

        ULTIMO_GRAFICO_GERADO = figura

        return (
            f"Grafico do tipo '{tipo_grafico}' gerado com sucesso e sera "
            f"exibido na interface, com '{coluna_categoria}' no eixo de "
            f"categorias e '{coluna_valor}' como valores."
        )

    except Exception as erro:
        return f"Nao foi possivel gerar o grafico: {erro}"

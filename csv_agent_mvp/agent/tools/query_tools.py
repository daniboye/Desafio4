"""
Módulo: agent/tools/query_tools.py

Aqui moram as "ferramentas" (tools) que o agente de IA pode chamar.

Um conceito-chave: o AGENTE (a IA) nunca calcula nada sozinho. Ele só
DECIDE qual dessas funções chamar e com quais parâmetros, a partir da
pergunta em português do usuário. Quem efetivamente soma, filtra e ordena
os dados é este código Python comum, usando a biblioteca pandas.

Isso é importante para confiabilidade: cálculos de soma, média, contagem
etc. feitos por uma IA "de cabeça" podem estar errados (é um dos problemas
mais conhecidos de LLMs). Fazendo o cálculo aqui, em código determinístico,
garantimos que o número está sempre correto.

Usamos o decorator @tool do LangChain, que transforma uma função Python
comum em algo que o agente consegue "enxergar" e chamar.
"""

from typing import Optional
from langchain_core.tools import tool
import pandas as pd


# Este dicionário guarda as tabelas atualmente carregadas na sessão do
# usuário. Ele é preenchido pela interface (Streamlit) assim que o upload
# do ZIP é processado, e é isso que permite que as tools "enxerguem" os
# dados sem que o agente precise passá-los a cada chamada.
_TABELAS_CARREGADAS: dict[str, pd.DataFrame] = {}


def definir_tabelas_ativas(tabelas: dict[str, pd.DataFrame]) -> None:
    """
    Chamada pela interface (app.py) assim que o usuário carrega um novo
    arquivo .ZIP, para "avisar" as tools quais tabelas estão disponíveis
    nesta sessão.
    """
    global _TABELAS_CARREGADAS
    _TABELAS_CARREGADAS = tabelas


def _obter_tabela(nome_tabela: str) -> pd.DataFrame:
    """
    Busca uma tabela pelo nome, com tolerância a pequenas diferenças de
    digitação (maiúsculas/minúsculas, espaços), já que o nome vem de uma
    decisão da IA e pode não ser character-by-character idêntico.
    """
    if nome_tabela in _TABELAS_CARREGADAS:
        return _TABELAS_CARREGADAS[nome_tabela]

    nome_normalizado = nome_tabela.strip().lower()
    for nome_real, df in _TABELAS_CARREGADAS.items():
        if nome_real.strip().lower() == nome_normalizado:
            return df

    nomes_disponiveis = ", ".join(_TABELAS_CARREGADAS.keys())
    raise ValueError(
        f"A tabela '{nome_tabela}' não foi encontrada. "
        f"Tabelas disponíveis: {nomes_disponiveis}"
    )


@tool
def consultar_dataframe(
    tabela: str,
    agrupar_por: Optional[str] = None,
    agregacao: Optional[str] = None,
    coluna_agregada: Optional[str] = None,
    ordenar_por: Optional[str] = None,
    ordem: str = "desc",
    limite: int = 10,
    filtro_coluna: Optional[str] = None,
    filtro_valor: Optional[str] = None,
) -> str:
    """
    Consulta uma tabela de dados, podendo filtrar, agrupar, agregar e
    ordenar os resultados. Use esta ferramenta para responder perguntas
    como "qual fornecedor teve o maior valor total", "quais os 5 produtos
    mais vendidos", "total de notas por estado" etc.

    Args:
        tabela: nome exato da tabela a consultar (ex.: "202401_NFs_Cabecalho").
        agrupar_por: nome da coluna para agrupar os dados (ex.: "UF DESTINATÁRIO").
            Deixe em branco se não for necessário agrupar.
        agregacao: tipo de cálculo a aplicar em cada grupo. Use um dos
            valores: "soma", "media", "contagem", "minimo", "maximo".
        coluna_agregada: nome da coluna numérica sobre a qual calcular a
            agregação (ex.: "VALOR NOTA FISCAL"). Obrigatório se
            'agregacao' for informado, exceto para "contagem".
        ordenar_por: nome da coluna pela qual ordenar o resultado final.
        ordem: "desc" para do maior para o menor, ou "asc" para o contrário.
        limite: número máximo de linhas a devolver (padrão: 10).
        filtro_coluna: nome de uma coluna para filtrar os dados antes de
            processar (ex.: "UF DESTINATÁRIO").
        filtro_valor: valor exato que a 'filtro_coluna' deve ter para a
            linha ser incluída (ex.: "RJ").

    Returns:
        Uma tabela de resultado em formato de texto (Markdown), pronta
        para ser apresentada ao usuário.
    """
    try:
        df = _obter_tabela(tabela).copy()

        if filtro_coluna and filtro_valor:
            if filtro_coluna not in df.columns:
                return f"Erro: a coluna '{filtro_coluna}' não existe na tabela '{tabela}'."
            df = df[df[filtro_coluna].astype(str).str.contains(filtro_valor, case=False, na=False)]
            if df.empty:
                return f"Nenhuma linha encontrada com '{filtro_coluna}' contendo '{filtro_valor}'."

        if agrupar_por:
            if agrupar_por not in df.columns:
                return f"Erro: a coluna '{agrupar_por}' não existe na tabela '{tabela}'."

            mapa_agregacoes = {
                "soma": "sum",
                "media": "mean",
                "contagem": "count",
                "minimo": "min",
                "maximo": "max",
            }
            agregacao_pandas = mapa_agregacoes.get(agregacao, "sum")

            if agregacao == "contagem" or coluna_agregada is None:
                resultado = df.groupby(agrupar_por).size().reset_index(name="contagem")
                coluna_resultado = "contagem"
            else:
                if coluna_agregada not in df.columns:
                    return f"Erro: a coluna '{coluna_agregada}' não existe na tabela '{tabela}'."
                resultado = (
                    df.groupby(agrupar_por)[coluna_agregada]
                    .agg(agregacao_pandas)
                    .reset_index()
                )
                coluna_resultado = coluna_agregada

            coluna_ordenacao = ordenar_por or coluna_resultado
            resultado = resultado.sort_values(
                by=coluna_ordenacao, ascending=(ordem == "asc")
            )
        else:
            resultado = df
            if ordenar_por:
                if ordenar_por not in df.columns:
                    return f"Erro: a coluna '{ordenar_por}' não existe na tabela '{tabela}'."
                resultado = resultado.sort_values(by=ordenar_por, ascending=(ordem == "asc"))

        resultado = resultado.head(limite)

        if resultado.empty:
            return "A consulta não retornou nenhum resultado."

        return resultado.to_markdown(index=False)

    except Exception as erro:
        return (
            f"Ocorreu um erro ao consultar os dados: {erro}. "
            f"Verifique se os nomes de tabela e coluna estão corretos."
        )


@tool
def resumo_estatistico(tabela: str, coluna: str) -> str:
    """
    Calcula estatísticas descritivas (soma, média, contagem, mínimo,
    máximo) de uma coluna numérica de uma tabela. Use para perguntas como
    "qual o valor médio das notas fiscais" ou "quantas notas foram emitidas".

    Args:
        tabela: nome exato da tabela a consultar.
        coluna: nome da coluna numérica a analisar (ex.: "VALOR NOTA FISCAL").

    Returns:
        Um texto com soma, média, contagem, mínimo e máximo da coluna.
    """
    try:
        df = _obter_tabela(tabela)

        if coluna not in df.columns:
            return f"Erro: a coluna '{coluna}' não existe na tabela '{tabela}'."

        serie_numerica = pd.to_numeric(df[coluna], errors="coerce")

        return (
            f"Estatísticas da coluna '{coluna}' na tabela '{tabela}':\n"
            f"- Contagem de linhas: {serie_numerica.count()}\n"
            f"- Soma: {serie_numerica.sum():.2f}\n"
            f"- Média: {serie_numerica.mean():.2f}\n"
            f"- Mínimo: {serie_numerica.min():.2f}\n"
            f"- Máximo: {serie_numerica.max():.2f}"
        )

    except Exception as erro:
        return f"Ocorreu um erro ao calcular as estatísticas: {erro}."

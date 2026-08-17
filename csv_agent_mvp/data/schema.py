"""
Módulo: data/schema.py

Depois que o loader.py transforma o .ZIP em tabelas (DataFrames), este
módulo descreve, EM TEXTO, o que existe dentro de cada tabela: nomes de
colunas, tipos de dado e alguns valores de exemplo.

Por quê isso é necessário? Porque a IA (o LLM) não "vê" o DataFrame do
jeito que o Python vê. A única forma de o modelo saber que existe uma
coluna chamada "VALOR NOTA FISCAL" ou que a tabela "202401_NFs_Cabecalho"
tem 100 linhas é... alguém contar isso a ele, em texto, dentro do prompt.

Esse texto gerado aqui (chamamos de "schema_context") é o que vai ser
injetado no prompt de sistema do agente, no próximo módulo.
"""

import pandas as pd


def _tipo_amigavel(tipo_pandas) -> str:
    """
    Traduz os tipos técnicos do pandas (int64, float64, object...) para
    palavras mais claras, que ajudam a IA a entender o tipo de dado sem
    precisar conhecer jargão de programação.
    """
    tipo_texto = str(tipo_pandas)
    if "int" in tipo_texto or "float" in tipo_texto:
        return "número"
    if "datetime" in tipo_texto:
        return "data/hora"
    if "bool" in tipo_texto:
        return "verdadeiro/falso"
    return "texto"


def _descrever_coluna(df: pd.DataFrame, nome_coluna: str, descricoes: dict[str, str]) -> str:
    """
    Monta uma linha de descrição para UMA coluna de UMA tabela, no formato:
      - NOME_DA_COLUNA (tipo): descrição do dicionário de dados (se houver).
        Exemplos de valores: valor1, valor2, valor3
    """
    tipo = _tipo_amigavel(df[nome_coluna].dtype)

    # Pega até 3 valores não vazios, sem repetir, para dar um exemplo real
    # à IA de como os dados daquela coluna se parecem.
    exemplos = (
        df[nome_coluna]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(3)
        .tolist()
    )
    exemplos_texto = ", ".join(exemplos) if exemplos else "(sem exemplos disponíveis)"

    descricao_extra = descricoes.get(nome_coluna, "")
    linha = f"  - {nome_coluna} ({tipo})"
    if descricao_extra:
        linha += f": {descricao_extra}"
    linha += f"\n    Exemplos de valores: {exemplos_texto}"

    return linha


def montar_schema_context(
    tabelas: dict[str, pd.DataFrame],
    dicionario_dados: dict[str, str] | None = None,
) -> str:
    """
    Função principal deste módulo.

    Parâmetros:
      tabelas: o dicionário {nome_tabela: DataFrame} vindo do loader.py.
      dicionario_dados: o dicionário {nome_coluna: descrição}, se existir.

    Devolve: uma única string de texto, pronta para ser inserida no prompt
    do agente, descrevendo todas as tabelas e colunas disponíveis.
    """
    descricoes = dicionario_dados or {}
    blocos_de_texto = []

    for nome_tabela, df in tabelas.items():
        linhas_da_tabela = [
            f"Tabela: \"{nome_tabela}\" ({df.shape[0]} linhas, {df.shape[1]} colunas)",
            "Colunas:",
        ]
        for nome_coluna in df.columns:
            linhas_da_tabela.append(_descrever_coluna(df, nome_coluna, descricoes))

        blocos_de_texto.append("\n".join(linhas_da_tabela))

    texto_completo = "\n\n".join(blocos_de_texto)

    # Adicionamos uma dica extra sobre como as tabelas de notas fiscais
    # costumam se relacionar entre si, quando os nomes sugerem isso.
    # Isso ajuda o agente a saber que pode "juntar" Cabecalho e Itens.
    nomes_tabelas = list(tabelas.keys())
    dica_relacionamento = ""
    if len(nomes_tabelas) > 1 and any("CHAVE DE ACESSO" in tabelas[t].columns for t in nomes_tabelas):
        dica_relacionamento = (
            "\n\nObservação: quando mais de uma tabela possui a coluna "
            "\"CHAVE DE ACESSO\", essa coluna identifica de forma única cada "
            "nota fiscal e pode ser usada para relacionar informações entre "
            "as tabelas (por exemplo, unir dados de cabeçalho da nota com "
            "os itens/produtos daquela mesma nota)."
        )

    return texto_completo + dica_relacionamento

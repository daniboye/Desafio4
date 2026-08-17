"""
Módulo: data/loader.py

Este arquivo cuida de UMA responsabilidade: pegar o arquivo .ZIP que o
usuário enviou e transformá-lo em algo que o Python consegue usar:
- um dicionário de tabelas (DataFrames do pandas), uma por CSV encontrado;
- opcionalmente, um dicionário de dados (a descrição das colunas), se o
  usuário incluiu esse arquivo no ZIP.

Não há nada de Inteligência Artificial aqui ainda — isso é só leitura e
organização de arquivos. É a base sobre a qual o agente vai "enxergar" os
dados depois.
"""

import io
import zipfile
import pandas as pd


class ErroDeCarga(Exception):
    """
    Criamos um tipo de erro próprio (uma "exceção customizada") para
    diferenciar problemas esperados (ex.: "não tem CSV no zip") de erros
    inesperados do programa. Isso deixa as mensagens de erro mais claras
    para quem for usar a aplicação.
    """
    pass


def _ler_csv_com_fallback(conteudo_bytes: bytes, nome_arquivo: str) -> pd.DataFrame:
    """
    Tenta ler um CSV testando primeiro o formato mais comum no Brasil
    (separador vírgula, codificação utf-8) e, se falhar, tenta variações
    comuns (encoding latin-1, separador ponto e vírgula).

    Recebe: os bytes brutos do arquivo e o nome (só para mensagens de erro).
    Devolve: um DataFrame do pandas (a "tabela" que o pandas entende).
    """
    tentativas = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin-1", "sep": ","},
        {"encoding": "latin-1", "sep": ";"},
    ]

    ultimo_erro = None
    for opcoes in tentativas:
        try:
            return pd.read_csv(io.BytesIO(conteudo_bytes), **opcoes)
        except Exception as erro:
            ultimo_erro = erro
            continue

    # Se nenhuma tentativa funcionou, avisamos com uma mensagem clara,
    # em vez de deixar o programa quebrar com um erro técnico confuso.
    raise ErroDeCarga(
        f"Não foi possível ler o arquivo '{nome_arquivo}' como CSV. "
        f"Verifique se o arquivo está no formato correto. Erro técnico: {ultimo_erro}"
    )


def carregar_zip(arquivo_zip_em_memoria) -> tuple[dict[str, pd.DataFrame], dict[str, str] | None]:
    """
    Função principal deste módulo.

    Parâmetro:
      arquivo_zip_em_memoria: o arquivo enviado pelo usuário (no Streamlit,
      isso vem do componente de upload — por enquanto, para efeitos de
      ensino, também aceitamos um caminho de arquivo no disco).

    Devolve DUAS coisas (uma "tupla"):
      1) um dicionário {nome_da_tabela: DataFrame} — por exemplo:
         {"202401_NFs_Cabecalho": <tabela com 100 notas fiscais>,
          "202401_NFs_Itens": <tabela com 565 itens>}
      2) um dicionário de dados {nome_coluna: descrição}, ou None se o
         usuário não enviou um dicionário de dados dentro do ZIP.
    """
    try:
        zip_obj = zipfile.ZipFile(arquivo_zip_em_memoria)
    except zipfile.BadZipFile:
        raise ErroDeCarga(
            "O arquivo enviado não é um .ZIP válido. Verifique se o "
            "arquivo não está corrompido e tente novamente."
        )

    tabelas: dict[str, pd.DataFrame] = {}
    dicionario_dados: dict[str, str] | None = None

    nomes_no_zip = [n for n in zip_obj.namelist() if not n.endswith("/")]

    if not nomes_no_zip:
        raise ErroDeCarga("O arquivo .ZIP está vazio.")

    for nome_interno in nomes_no_zip:
        nome_minusculo = nome_interno.lower()

        # Ignoramos arquivos de sistema que o macOS/Windows às vezes
        # incluem automaticamente dentro de um zip (ex.: __MACOSX, .DS_Store)
        if "__macosx" in nome_minusculo or nome_minusculo.endswith(".ds_store"):
            continue

        conteudo = zip_obj.read(nome_interno)

        eh_dicionario_de_dados = (
            "dicionario" in nome_minusculo or "dictionary" in nome_minusculo
        )

        if nome_minusculo.endswith(".csv") and eh_dicionario_de_dados:
            dicionario_dados = _ler_dicionario_de_dados(conteudo, nome_interno)

        elif nome_minusculo.endswith(".csv"):
            nome_tabela = nome_interno.rsplit("/", 1)[-1].replace(".csv", "")
            tabelas[nome_tabela] = _ler_csv_com_fallback(conteudo, nome_interno)

        # Outros tipos de arquivo dentro do zip (ex.: .txt, .xlsx) são
        # simplesmente ignorados nesta primeira versão do MVP.

    if not tabelas:
        raise ErroDeCarga(
            "Nenhum arquivo .csv foi encontrado dentro do .ZIP enviado. "
            "Envie um .ZIP contendo pelo menos um arquivo .csv."
        )

    return tabelas, dicionario_dados


def _ler_dicionario_de_dados(conteudo_bytes: bytes, nome_arquivo: str) -> dict[str, str]:
    """
    Lê o CSV do dicionário de dados e o transforma em um dicionário Python
    simples: {"NOME DA COLUNA": "descrição da coluna"}.

    Convenção assumida: o dicionário de dados tem pelo menos duas colunas,
    a primeira com o nome da coluna original e a segunda com a descrição.
    Se o formato for diferente, apenas avisamos e seguimos sem quebrar o
    programa (o dicionário de dados é recomendável, não obrigatório).
    """
    try:
        df_dicionario = _ler_csv_com_fallback(conteudo_bytes, nome_arquivo)
        primeira_coluna = df_dicionario.columns[0]
        segunda_coluna = df_dicionario.columns[1]
        return dict(zip(df_dicionario[primeira_coluna], df_dicionario[segunda_coluna]))
    except Exception:
        # Não interrompemos a carga por causa do dicionário de dados:
        # ele enriquece as respostas do agente, mas não é essencial.
        return {}

import os
from functions import carregar_dados, gerar_graficos_eda, relatorio_estatistico

CAMINHO_BASE = os.path.join(
    "projeto", "data", "E Commerce Dataset - E Comm.csv" # caminho do arquivo
)
PASTA_GRAFICOS = os.path.join("projeto", "graficos")


def main():
    print("==================================================")
    print("FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("==================================================")

    df = carregar_dados(CAMINHO_BASE) # pego df que foi refinado e estava na memória e salvo em df e assim ser utilzado
    relatorio_estatistico(df)
    gerar_graficos_eda(df, pasta_saida=PASTA_GRAFICOS)
    print(df.skew(numeric_only=True)) # assimetria > 1 ou < -1: usar mediana (resistente a cauda longa e outliers)


main()
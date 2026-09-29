import os
from functions import carregar_dados, criar_feature_cashback_por_pedido, gerar_graficos_eda, relatorio_estatistico, gerar_graficos_outliers, tratar_duplicadas, tratar_nulos, tratar_outliers_clipping

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

    print("\n==================================================")
    print("FASE 2: TRATAMENTO E LIMPEZA (DATA PREP)")
    print("==================================================")
    # 1. Duplicadas
    df_sem_duplicadas = tratar_duplicadas(df)

    # 2. Imputação de Nulos via Mediana
    df_sem_nulos = tratar_nulos(df_sem_duplicadas)

    # 3. Análise visual de Outliers
    gerar_graficos_outliers(df_sem_nulos, pasta_saida=PASTA_GRAFICOS)

    # 4. Clipping em colunas com distâncias e durações extremas
    colunas_outliers = ["WarehouseToHome", "DaySinceLastOrder"]
    df_preparado = tratar_outliers_clipping(
        df_sem_nulos, colunas=colunas_outliers
    )

    print("\n==================================================")
    print("FASE 3: FEATURE ENGINEERING (COLUNA CALCULADA)")
    print("==================================================")
    df_final_features = criar_feature_cashback_por_pedido(df_preparado)
    print(
        f"[INFO] Dataset pronto para split e escalonamento. Total de colunas: {df_final_features.shape[1]}"
    )

main()
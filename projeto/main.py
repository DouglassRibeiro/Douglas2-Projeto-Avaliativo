import os
from functions import (
    avaliar_e_plotar_modelos_finais,
    carregar_dados,
    criar_feature_cashback_por_pedido,
    escalonar_dados_knn,
    gerar_graficos_eda,
    gerar_graficos_outliers,
    gerar_graficos_overfitting,
    otimizar_arvore,
    otimizar_knn,
    preparar_features_encoding,
    relatorio_estatistico,
    split_estratificado_balanceado,
    tratar_duplicadas,
    tratar_nulos,
    tratar_outliers_clipping,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
    df_fe = criar_feature_cashback_por_pedido(df_preparado)
    print(
        f"[INFO] Dataset pronto para split e escalonamento. Total de colunas: {df_fe.shape[1]}"
    )

    print("\n==================================================")
    print("FASE 4: SEPARAÇÃO, BALANCEAMENTO E ESCALONAMENTO SEGURO")
    print("==================================================")
    X, y = preparar_features_encoding(df_fe)

    # Split estratificado + SMOTE restrito ao treino
    X_train_res, X_test, y_train_res, y_test = split_estratificado_balanceado(
        X, y, test_size=0.20, random_state=42
    )

    # Escalonamento apenas para o KNN
    X_train_knn, X_test_knn, scaler = escalonar_dados_knn(
        X_train_res, X_test
    )

    print("\n[CHECKPOINT] Dados preparados para a Fase 5 (Modelagem):")
    print(f" -> Conjunto de Treino para Árvore (não escalonado): {X_train_res.shape}")
    print(f" -> Conjunto de Treino para KNN (escalonado): {X_train_knn.shape}")
    print(f" -> Conjunto de Teste real (intocado): {X_test.shape}")

    print("\n==================================================")
    print("FASE 5: MODELAGEM E VALIDAÇÃO (O DESAFIO DO OVERFITTING)")
    print("==================================================")
    df_res_knn = otimizar_knn(
        X_train_knn,
        y_train_res,
        X_test_knn,
        y_test,
        k_valores=[3, 5, 7, 9],
    )
    df_res_arvore = otimizar_arvore(
        X_train_res,
        y_train_res,
        X_test,
        y_test,
        depth_valores=[3, 5, 7, None],
    )
    gerar_graficos_overfitting(
        df_res_knn, df_res_arvore, pasta_saida=PASTA_GRAFICOS
    )

    print("\n==================================================")
    print("FASE 6: AVALIAÇÃO E VEREDITO DE NEGÓCIOS")
    print("==================================================")
    # Treina os melhores candidatos selecionados para avaliação final
    melhor_knn = KNeighborsClassifier(n_neighbors=3)
    melhor_knn.fit(X_train_knn, y_train_res)

    melhor_arvore = DecisionTreeClassifier(max_depth=7, random_state=42)
    melhor_arvore.fit(X_train_res, y_train_res)

    avaliar_e_plotar_modelos_finais(
        melhor_knn,
        X_test_knn,
        melhor_arvore,
        X_test,
        y_test,
        pasta_saida=PASTA_GRAFICOS,
    )
    
main()
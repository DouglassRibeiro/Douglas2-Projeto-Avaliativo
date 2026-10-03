from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Fase 1: Analise Exploratória de Dados (EDA)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def carregar_dados(caminho_csv: str) -> pd.DataFrame: # Carrega o dataset e exibe a dimensão inicial.
    caminho = Path(caminho_csv)
    if not caminho.exists(): # se e somente se o caminho fisíco não existir
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}") # exibi erro impedindo arquivos inexistentes, encerrando todo processamento.
    df = pd.read_csv(caminho) 
    print(
        f"[CARREGAMENTO] Linhas: {df.shape[0]} | Colunas: {df.shape[1]}"
    )
    return df # retorne a cópia do csv.


def relatorio_estatistico(df: pd.DataFrame) -> None: # Exibe tipos de dados, contagem de nulos e o sumário estatístico descritivo.
    print("\n--- 1. ESTRUTURA GERAL DOS DADOS ---")
    print(df.info()) # quantida de linhas total, número de colunas, nome de cada coluna, quantidade de valores não nulos e tipo

    print("\n--- 2. CONTAGEM DE VALORES NULOS POR COLUNA ---")
    nulos = df.isnull().sum() # contagem de nulos
    print(nulos[nulos > 0])

    print("\n--- 3. SUMÁRIO ESTATÍSTICO DESCRITIVO ---")
    print(df.describe().T) # .T ao invés de linhas mostra colunas, as quebra das porcentagem já são definidas por padrão.



def gerar_graficos_eda(df: pd.DataFrame, pasta_saida: str = "graficos") -> None: # 3 Gráficos.
    sns.set_theme(style="whitegrid")
    caminho_pasta = Path(pasta_saida)
    caminho_pasta.mkdir(parents=True, exist_ok=True)

    # Gráfico 1: Histograma de Distribuição (Tenure)
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Tenure"], kde=True, color="#2b5c8f", bins=30) # "bins=" blocos verticais representando a quantidade de clíentes em cada faixa e "kde=" uma linha suave demostrando a tedência
    plt.title("Distribuição do Tempo de Relacionamento (Tenure)")
    plt.xlabel("Meses de Relacionamento (Tenure)")
    plt.ylabel("Frequência de Clientes")
    plt.tight_layout()
    caminho_g1 = caminho_pasta / "eda_01_distribuicao_tenure.png"
    plt.savefig(caminho_g1, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_g1}")

    # Gráfico 2: Desbalanceamento do Churn (com hue ajustado para evitar warning) - barras de contagem
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        x="Churn",
        hue="Churn",
        data=df,
        palette=["#2ecc71", "#e74c3c"],
        legend=False,
    ) # mapeia a contagem bruta de registros

    plt.title("Proporção da Variável Alvo: Churn (0 = Retido, 1 = Evasão)")
    plt.xlabel("Status de Churn")
    plt.ylabel("Quantidade de Clientes")

    total = len(df)
    for p in ax.patches: # proporção relativa
        altura = p.get_height()
        porcentagem = f"{(altura / total) * 100:.1f}%"
        ax.annotate(
            f"{altura}\n({porcentagem})",
            (p.get_x() + p.get_width() / 2.0, altura / 2),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            weight="bold",
        )

    plt.tight_layout()
    caminho_g2 = caminho_pasta / "eda_02_desbalanceamento_churn.png"
    plt.savefig(caminho_g2, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_g2}")

    # Gráfico 3: Heatmap de Correlação de Pearson - mapa de calor menssurando a relação
    plt.figure(figsize=(12, 8))
    colunas_numericas = df.select_dtypes(include=["number"]).drop(
        columns=["CustomerID"]
    ) # quero apenas colunas númericas

    matriz_corr = colunas_numericas.corr(method="pearson") # correlaçaõ -1 +1

    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
    ) # intencidade varia de acordo com a relação 

    plt.title("Mapa de Calor: Correlação Linear de Pearson")
    plt.tight_layout()
    caminho_g3 = caminho_pasta / "eda_03_correlacao_pearson.png"
    plt.savefig(caminho_g3, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_g3}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Fase 2: Tratamento e Limpeza (Data Prep)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def tratar_duplicadas(df: pd.DataFrame) -> pd.DataFrame: # Verifica e remove linhas duplicadas para evitar redundância e viés.
    total_duplicadas = df.duplicated().sum() # .duplicated olha se o DataFrame possui linhas duplicadas, .sum() soma isso depois
    print(
        f"\n[DATA PREP] Linhas duplicadas encontradas no dataset: {total_duplicadas}"
    )

    if total_duplicadas > 0:
        df_limpo = df.drop_duplicates().copy() # .drop_duplicates apaga cópias duplicadas | "SettingWithCopyWarning" .copy - um novo local independente na meméria é alocado envitando aleteração no csv original
        print(
            f"[DATA PREP] Duplicadas removidas. Total de linhas atual: {len(df_limpo)}" # remoção de duplicidade é ecenssial para treinamento de um modelo
        )
        return df_limpo

    print("[DATA PREP] Nenhuma linha duplicada detectada.")
    return df.copy()

def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame: # Aplica imputação pela MEDIANA nas colunas numéricas com dados ausentes.
    df_imputado = df.copy()
    colunas_com_nulos = df_imputado.columns[ # colunas_com_nulos recebe apenas os nulos
        df_imputado.isnull().any() # isnull() gera uma tabela avisando onde esta vazio verdadeiro, depois any() confirma onde é verdadeiro nessa tabela. 
    ].tolist() # separando o que for confirmado em sua lista

    print("\n--- TRATAMENTO DE VALORES NULOS VIA MEDIANA ---")
    for coluna in colunas_com_nulos: # roda o total de colunas_com_nulos
        mediana_valor = df_imputado[coluna].median() # pega o valor mediano apresentado naquela coluna
        qtd_nulos = df_imputado[coluna].isnull().sum() # quantidade de nulos apresentados
        df_imputado[coluna] = df_imputado[coluna].fillna(mediana_valor) # vasculha a coluna e sempre que encontra uma célula vazia insere a mediana geral.
        print(
            f"Coluna '{coluna}': {qtd_nulos} nulos imputados com a mediana = {mediana_valor}"
        )

    return df_imputado

def gerar_graficos_outliers(df: pd.DataFrame, pasta_saida: str = "graficos") -> None: # Plota boxplots das variáveis mais propensas a outliers para fundamentar o tratamento.
    caminho_pasta = Path(pasta_saida)
    caminho_pasta.mkdir(parents=True, exist_ok=True)

    colunas_foco = ["WarehouseToHome", "DaySinceLastOrder", "Tenure"] 

    plt.figure(figsize=(10, 5)) # matplotlib fica por conta da parte gráfica
    sns.boxplot(data=df[colunas_foco], palette="Set2") # mediana, seaborn cuida dos calculos
    plt.title("Identificação de Outliers via Boxplot")
    plt.ylabel("Valores")
    plt.tight_layout()

    caminho_img = caminho_pasta / "prep_01_boxplots_outliers.png"
    plt.savefig(caminho_img, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_img}")

def tratar_outliers_clipping(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame: # Aplica clipping (capping) nos limites interquartis (IQR 1.5x) para colunas com discrepâncias extremas. 
    df_tratado = df.copy()
    print("\n--- TRATAMENTO DE OUTLIERS VIA CLIPPING (IQR) ---")
    for col in colunas:
        q1 = df_tratado[col].quantile(0.25)
        q3 = df_tratado[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        # Aplicar recorte sem descartar clientes
        valores_antes = (
            (df_tratado[col] < limite_inferior)
            | (df_tratado[col] > limite_superior)
        ).sum()
        df_tratado[col] = df_tratado[col].clip(lower=limite_inferior, upper=limite_superior)
        print(f"Coluna '{col}': {valores_antes} valores ajustados aos limites [{limite_inferior:.1f}, {limite_superior:.1f}]")

    return df_tratado

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Fase 3: Feature Engineering (Coluna Calculada)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def criar_feature_cashback_por_pedido(df: pd.DataFrame) -> pd.DataFrame: # cashback_por_pedido = CashbackAmount / OrderCount
   
    df_fe = df.copy()

    # Aplica validação prévia de nulos e proteção contra divisão por zero para evitar contaminação da base com valores inf ou NaN

    # Validação de integridade estatística
    if (
        df_fe["OrderCount"].isnull().any()
        or df_fe["CashbackAmount"].isnull().any()
    ):
        raise ValueError( # valores nulos também não são aceitos
            "Detectados valores nulos nas variáveis de cálculo. Trate os nulos antes do cálculo."
        )

    # Proteção de negócio caso existisse contagem zerada de pedidos
    if (df_fe["OrderCount"] == 0).any(): # Modelos KNN quebram em dados com divisão por 0
        print(
            "[AVISO] Pedidos iguais a 0 detectados. Ajustando para 1 para evitar divisão por zero."
        )
        df_fe["OrderCount"] = df_fe["OrderCount"].replace(0, 1)

    # Cálculo da taxa
    df_fe["cashback_por_pedido"] = (
        df_fe["CashbackAmount"] / df_fe["OrderCount"]
    )

    print("\n--- FASE 3: FEATURE ENGINEERING ---")
    print(
        f"[NOVA FEATURE] 'cashback_por_pedido' criada. Média: {df_fe['cashback_por_pedido'].mean():.2f} | Mediana: {df_fe['cashback_por_pedido'].median():.2f}" # calculos da média e a mediana
    )

    return df_fe

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Fase 4: Separação, Balanceamento e Escalonamento Seguro
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def padronizar_categorias(df: pd.DataFrame) -> pd.DataFrame: # Padroniza nomeclaturas duplicadas na mesma colula.
    df_padrao = df.copy()
    # categorias que representam a mesma entidade não devem ter colunas separadas
    mapeamentos = { 
        "PreferredLoginDevice": {"Phone": "Mobile Phone"},
        "PreferredPaymentMode": {
            "CC": "Credit Card",
            "COD": "Cash on Delivery",
        },
        "PreferedOrderCat": {"Mobile": "Mobile Phone"},
    }
    for col, correcoes in mapeamentos.items():
        if col in df_padrao.columns:
            df_padrao[col] = df_padrao[col].replace(correcoes)

    return df_padrao


def preparar_features_encoding(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: # One-Hot Encoding - convertendo variaveis textuais em dados binário
    # Não esquecer de Descarta CustomerID - não estamos analizando ids especificos, e sim o valor que o tipo de cliente representa apredendo padrões que levam a isso
    df_modelo = df.copy()

    # Descartar chave primária sem valor preditivo
    if "CustomerID" in df_modelo.columns:
        df_modelo = df_modelo.drop(columns=["CustomerID"]) # descarte CustomerID, aprenda comportamentos e não IDs

    df_modelo = padronizar_categorias(df_modelo) # categorias unicas

    # Separação X e y - perguntas X | respostas y
    X = df_modelo.drop(columns=["Churn"])
    y = df_modelo["Churn"]

    # One-Hot Encoding seguro
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int) # Pega colunas geradas

    print("\n--- FASE 4: ENCODING E PREPARAÇÃO ---")
    print(
        f"[ENCODING] Features preditoras: {X_encoded.shape[1]} colunas geradas."
    )

    return X_encoded, y


def split_estratificado_balanceado(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42) -> tuple: # Divide os dados em Treino e Teste (80% | 20%) preservando a proporção de classes, e aplica o SMOTE exclusivamente no treino
    # organizar proteção absoluta contra Data Leakage.
    # random_state: int = 42 é nível de sorteio garantindo reprodutibilidade dos resultados

    # 1. Divisão Estratificada - separacao para treino
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # stratify=y força o sorteio a manter a exata mesma taxa de Churn (rotatividade de clientes em ambos os conjuntos, os 16,8%)
    )

    print(
        f"[SPLIT] Treino original: {X_train.shape[0]} amostras | Teste original: {X_test.shape[0]} amostras"
    )
    print(f"[SPLIT] Distribuição y_train: {dict(y_train.value_counts())}")
    print(
        f"[SPLIT] Distribuição y_test (intocada): {dict(y_test.value_counts())}"
    )

    # 2. Reamostragem estritamente no treino (Anti-Leakage)
    smote = SMOTE(random_state=random_state) # os novos exemplos sintéticos gerados com base no Churn
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # garante que o SMOTE seja aplicado estritamente e apenas nos dados de treino

    print(
        f"[SMOTE] Treino balanceado: {X_train_res.shape[0]} amostras (Classe 0: {(y_train_res == 0).sum()} | Classe 1: {(y_train_res == 1).sum()})"
    )

    return X_train_res, X_test, y_train_res, y_test


def escalonar_dados_knn(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    
    scaler = StandardScaler()
    X_train_knn = scaler.fit_transform(X_train)
    X_test_knn = scaler.transform(X_test)

    print(
        "[SCALER] Standard Scaler aplicado com segurança exclusivamente para uso no KNN."
    )

    return X_train_knn, X_test_knn, scaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Fase 5: Modelagem e Validação (O Desafio do Overfitting)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Treina e avalia o KNN para múltiplos valores de K em Treino e Teste simultaneamente.
def otimizar_knn(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series, k_valores: list[int] = [3, 5, 7, 9]) -> pd.DataFrame: 

    resultados = []

    print("\n--- EXPERIMENTAÇÃO KNN: MONITORAMENTO DE OVERFITTING ---")

    for k in k_valores: # laço de repetição iterando pela lista de hiperparâmetros exigida.
        modelo = KNeighborsClassifier(n_neighbors=k) # Instancia o estimador definindo a quantidade de vizinhos que terão direito a voto na classificação.
        modelo.fit(X_train, y_train) # Carrega o espaço vetorial com os dados de treino escalonados

        # Previsões em treino e teste
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        # Calculo - taixa de acerto global
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)


        # O termômetro do Overfitting.
        gap_acc = (acc_train - acc_test) * 100 # Se o treino estiver em 99% e o teste em 85%, o gap é de 14%, evidenciando sobreajuste severo.

        resultados.append(
            {
                "Parametro": f"K={k}",
                "Valor": k,
                "Acc_Treino": acc_train,
                "Acc_Teste": acc_test,
                "F1_Treino": f1_train,
                "F1_Teste": f1_test,
                "Gap_Overfitting_Acc(%)": gap_acc,
            }
        )

    df_res = pd.DataFrame(resultados)
    print(df_res.to_string(index=False))
    return df_res

# Treina e avalia a Arvore de decisão para múltiplos valores de K em Treino e Teste simultaneamente.
def otimizar_arvore(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, depth_valores: list = [3, 5, 7, None] ) -> pd.DataFrame: 
    
    resultados = []

    print("\n--- EXPERIMENTAÇÃO ÁRVORE: MONITORAMENTO DE OVERFITTING ---")
    for depth in depth_valores:
        nome_param = f"max_depth={depth}"
        modelo = DecisionTreeClassifier(max_depth=depth, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)

        gap_acc = (acc_train - acc_test) * 100

        resultados.append(
            {
                "Parametro": nome_param,
                "Valor": str(depth),
                "Acc_Treino": acc_train,
                "Acc_Teste": acc_test,
                "F1_Treino": f1_train,
                "F1_Teste": f1_test,
                "Gap_Overfitting_Acc(%)": gap_acc,
            }
        )

    df_res = pd.DataFrame(resultados)
    print(df_res.to_string(index=False))
    return df_res

def gerar_graficos_overfitting( df_knn: pd.DataFrame, df_arvore: pd.DataFrame, pasta_saida: str = "graficos" ) -> None: # Plota as curvas de acurácia de Treino vs. Teste para comprovar visualmente o diagnóstico de overfitting.
    caminho_pasta = Path(pasta_saida)
    caminho_pasta.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva KNN
    axes[0].plot(
        df_knn["Valor"],
        df_knn["Acc_Treino"],
        marker="o",
        label="Treino (Escalonado)",
        color="#e74c3c",
    )
    axes[0].plot(
        df_knn["Valor"],
        df_knn["Acc_Teste"],
        marker="s",
        label="Teste (Generalização)",
        color="#2b5c8f",
    )
    axes[0].set_title("KNN: Efeito do Hiperparâmetro K no Overfitting")
    axes[0].set_xlabel("Número de Vizinhos (K)")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(True)

    # Curva Árvore
    axes[1].plot(
        df_arvore["Valor"],
        df_arvore["Acc_Treino"],
        marker="o",
        label="Treino (Balanceado)",
        color="#e74c3c",
    )
    axes[1].plot(
        df_arvore["Valor"],
        df_arvore["Acc_Teste"],
        marker="s",
        label="Teste (Generalização)",
        color="#27ae60",
    )
    axes[1].set_title(
        "Árvore de Decisão: Efeito de max_depth no Overfitting"
    )
    axes[1].set_xlabel("Profundidade Máxima (max_depth)")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    caminho_img = caminho_pasta / "mod_01_curvas_overfitting.png"
    plt.savefig(caminho_img, dpi=300)
    plt.close()
    print(f"\n[GRÁFICO SALVO] {caminho_img}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Fase 6: Avaliação e Veredito de Negócios
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Gera classification reports detalhados e plota as matrizes de confusão dos dois modelos campeões.
def avaliar_e_plotar_modelos_finais( modelo_knn, X_test_knn: np.ndarray, modelo_arvore, X_test_arvore: pd.DataFrame, y_test: pd.Series, pasta_saida: str = "graficos" ) -> None:
    caminho_pasta = Path(pasta_saida)
    caminho_pasta.mkdir(parents=True, exist_ok=True)

    # 1. Previsões no conjunto de teste intocado
    y_pred_knn = modelo_knn.predict(X_test_knn)
    y_pred_tree = modelo_arvore.predict(X_test_arvore)

    print("\n==================================================")
    print("RELATÓRIO DE CLASSIFICAÇÃO: MELHOR KNN (K=3)")
    print("==================================================")
    print(classification_report(y_test, y_pred_knn, digits=4))

    print("\n==================================================")
    print("RELATÓRIO DE CLASSIFICAÇÃO: ÁRVORE REGULARIZADA (max_depth=7)")
    print("==================================================")
    print(classification_report(y_test, y_pred_tree, digits=4))

    # 2. Plotagem lado a lado das Matrizes de Confusão
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_knn,
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Matriz de Confusão: KNN (K=3)")
    axes[0].set_xlabel("Previsão do Modelo")
    axes[0].set_ylabel("Valor Real")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_tree,
        cmap="Greens",
        ax=axes[1],
        colorbar=False,
    )
    axes[1].set_title("Matriz de Confusão: Árvore (max_depth=7)")
    axes[1].set_xlabel("Previsão do Modelo")
    axes[1].set_ylabel("Valor Real")

    plt.tight_layout()
    caminho_img = caminho_pasta / "mod_02_matrizes_confusao.png"
    plt.savefig(caminho_img, dpi=300)
    plt.close()
    print(f"\n[GRÁFICO SALVO] {caminho_img}")
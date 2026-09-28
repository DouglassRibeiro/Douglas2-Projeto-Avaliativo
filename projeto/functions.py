from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

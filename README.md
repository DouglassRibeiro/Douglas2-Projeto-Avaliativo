**```a princípio já que estou trabalhando em .py os gráficos serão armazenados na pasta projeto/graficos/...```**
---

- **```if``` Recebe o endereço do arquivo em formato de texto, converte em Path e confirma se o caminho existe, caso não, adiciona uma menssagem de erro sem travar todo processamento.**
- **```return df``` retorna uma cópia pra memória**

```python
def carregar_dados(caminho_csv: str) -> pd.DataFrame: # Carrega o dataset e exibe a dimensão inicial.
    caminho = Path(caminho_csv)
    if not caminho.exists(): # apenas verifica se existe na memória.
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}")
    df = pd.read_csv(caminho)
    print(
        f"[CARREGAMENTO] Linhas: {df.shape[0]} | Colunas: {df.shape[1]}"
    )
    return df
```

---

- **```df.info``` > exibi tipos de todas colunas**
- **```df.isnull().sum()``` > isola e soma somente os valores nulos**
- **```df.describe().T``` > gera uma estatistica descritiva completa. Exibindo colunas ao invés de linhas**

```python
def relatorio_estatistico(df: pd.DataFrame) -> None: # Exibe tipos de dados, contagem de nulos e o sumário estatístico descritivo.
    print("\n--- 1. ESTRUTURA GERAL DOS DADOS ---")
    print(df.info())

    print("\n--- 2. CONTAGEM DE VALORES NULOS POR COLUNA ---")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0])

    print("\n--- 3. SUMÁRIO ESTATÍSTICO DESCRITIVO ---")
    print(df.describe().T)

```

---

- **Gráficos:**

```python
def gerar_graficos_eda(df: pd.DataFrame, pasta_saida: str = "graficos") -> None: # 3 Gráficos.
    sns.set_theme(style="whitegrid")
    caminho_pasta = Path(pasta_saida)
    caminho_pasta.mkdir(parents=True, exist_ok=True)

    # Gráfico 1: Histograma de Distribuição (Tenure)
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Tenure"], kde=True, color="#2b5c8f", bins=30)
    plt.title("Distribuição do Tempo de Relacionamento (Tenure)")
    plt.xlabel("Meses de Relacionamento (Tenure)")
    plt.ylabel("Frequência de Clientes")
    plt.tight_layout()
    caminho_g1 = caminho_pasta / "eda_01_distribuicao_tenure.png"
    plt.savefig(caminho_g1, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_g1}")
```

![Distribuição de Tenure](projeto/graficos/eda_01_distribuicao_tenure.png)

#
#

```python
    # Gráfico 2: Desbalanceamento do Churn (com hue ajustado para evitar warning)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        x="Churn",
        hue="Churn",
        data=df,
        palette=["#2ecc71", "#e74c3c"],
        legend=False,
    )
    plt.title("Proporção da Variável Alvo: Churn (0 = Retido, 1 = Evasão)")
    plt.xlabel("Status de Churn")
    plt.ylabel("Quantidade de Clientes")

    total = len(df)
    for p in ax.patches:
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
```

![Desbalanceamento do Churn](projeto/graficos/eda_02_desbalanceamento_churn.png)

#
#

```python

    # Gráfico 3: Heatmap de Correlação de Pearson
    plt.figure(figsize=(12, 8))
    colunas_numericas = df.select_dtypes(include=["number"]).drop(
        columns=["CustomerID"]
    )
    matriz_corr = colunas_numericas.corr(method="pearson")

    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
    )
    plt.title("Mapa de Calor: Correlação Linear de Pearson")
    plt.tight_layout()
    caminho_g3 = caminho_pasta / "eda_03_correlacao_pearson.png"
    plt.savefig(caminho_g3, dpi=300)
    plt.close()
    print(f"[GRÁFICO SALVO] {caminho_g3}")
```

![Matriz de Correlação de Pearson](projeto/graficos/eda_03_correlacao_pearson.png)


1. **Fica visivel nessa analise que há picos incomuns tendo uma descrepancia muito grande em o que é retido, e o que há de evasão. É mostrado um desbalanceamento Severo: A base conta com cerca de 83,2% de clientes ativos (classe 0) e apenas 16,8% evadidos (classe 1). Treinar modelos diretamente sem balanceamento fará o algoritmo priorizar a classe majoritária, gerando falsos negativos críticos.**


2. **Dados Ausentes e Assimetria: Sete variáveis numéricas possuem valores nulos (Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder). Colunas como WarehouseToHome = 1.62 e CouponUsed = 2.55 apresentam cauda longa à direita e presença de valores extremos. Portanto, a mediana é a técnica estatisticamente recomendada para a imputação, pois a média sofreria distorção pelos outliers, sendo drasticamente alterada pelo maior valor.**


3. **Multicolinearidade e Relação Linear: No mapa de correlação, CustomerID não possui valor preditivo (deve ser descartado é apenas o ID), enquanto variáveis comportamentais como Complain e Tenure demonstram forte correlação com o Churn.** 

#
#

- **Removendo duplicidades organizando o mesmo peso há diferentes clientes, garantindo que o algoritimo de atenção igual**
- **Pra garantir que ocorra aprendizado a remoção de duplicidade é essencial**
```python
    def tratar_duplicadas(df: pd.DataFrame) -> pd.DataFrame: # Verifica e remove linhas duplicadas para evitar redundância e viés.
        total_duplicadas = df.duplicated().sum() 
        # .duplicated olha se o DataFrame possui linhas duplicadas 
        # .sum() soma isso depois, total_suplicadas recebe o total  
        print(
            f"\n[DATA PREP] Linhas duplicadas encontradas no dataset: {total_duplicadas}"
        )

        if total_duplicadas > 0:
            df_limpo = df.drop_duplicates().copy() 
            # .drop_duplicates apaga cópias duplicadas 
            # "SettingWithCopyWarning" .copy - um novo local independente na meméria é alocado envitando aleteração no csv original
            print(
                f"[DATA PREP] Duplicadas removidas. Total de linhas atual: {len(df_limpo)}" # remoção de duplicidade é ecenssial para treinamento de um modelo
            )
            return df_limpo # remoção de duplicidade é ecencial para treinamento de um modelo

        print("[DATA PREP] Nenhuma linha duplicada detectada.")
        return df.copy()
```

---

- **Por que usar mediana e não média?**
- **Foi confirmado altos valores longe do normal (cauda longa à direita), apenas a divisão traria uma média fora do comum**
- **O que deixa a analise proporcional da mediana como a melhor escolha tendo um padrão melhor a ser ponderado**

```python
    def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame: # aplica imputação pela MEDIANA nas colunas numéricas com dados ausentes.
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
```
- **Nota:** 
    - Adicionar a mediana geral onde nulos são encontrados seria um problema se o cliente especifico já não fosse tão ativo.
    - A mediana geral da base para OrderCount é 2.0.
    - Ao imputar 2.0, estamos atribuindo 2 pedidos para esse cliente, o que gera uma distorção controlada, mas ainda assim artificial.
    - Se esse cliente fosse alguém que comprou apenas 1 vez (ou estava inativo há muito tempo), estaríamos dobrando seu volume transacional teórico.
        - Esse não foi um requisito levantado pelo projeto, vou aceitar um erro residual para proteger a escala global dos algortimos preditivos.

---

- **Demonstrando a presença de valores discrepantes na base via boxplot**
- **De resto resume visualmente como os dados estão espalhados**
- **Deixando claro na parte colorida onde é abrangido os 50% e sua mediana representada pela linha horizontal**
- **Sua antena marca a variação aceitavel**

![Boxplot (Diagrama de Caixa)](projeto/graficos/prep_01_boxplots_outliers.png)

---

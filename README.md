# Detecção de Fraudes em Transações Bancárias

Projeto de Machine Learning para identificar padrões que diferenciam transações fraudulentas de transações legítimas em dados de cartão de crédito.

## Objetivo

O objetivo desta análise é identificar padrões que diferenciam transações fraudulentas de transações legítimas, utilizando técnicas de classificação, balanceamento declasses e explicabilidade de modelos (SHAP).

## Dataset 

O dataset utilizado contém transações realizadas por cartões de crédito, carregado diretamente via `pandas.read_csv` a partir de: 

```
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

### Principais colunas: 

| Coluna | Descrição |
|---|---|
| `Time` | Tempo (em segundos) desde a primeira transação registrada |
| `V1`a `V28` | Variáveis anoniminadas via PCA (para preservas a privacidade) |
| `Amount` | Valor da transação | 
| `Class` | Rótulo alto - `0` = transação legítima, `1` = fraude |

**Dataset extremamente desbalanceado**: ~99,83% das transações são legítimas e apenas ~0,17% são fraudes. Esse desbalanceamento é o principal desafio do problema, pois um modelo "ingênuo" (sempre prevê "não fraude") teria altíssima acurácia sem nenhuma utilidade real.

## Etapas do projeto

1. **Coleta de dados** - carregamento do dataset e análise inicial (`df.info()`, `df.columns`, distribuição da classe alvo).
2. **Feature Enginnering**
    - `Amount_log`: transformação logarítma (`np.log1p`) sobre `Amount`, para suavizar a alta variação nos valores das transações.
    - `Amount_scaled`: padronização (`StandardScaler`) do valor da transação.
3. **Separação treino/teste** - `train_test_split` com `test_size=0.3`, `stratify=y`(mantém a proporção de fraudes em treino e teste) e `random_state=42`.
4. **Modelagem e avaliação**
    - Regressão Logística (baseline)
    - Curva ROC e AUC
    - Curva Precision-Recall
    - Balanceamento de classes: undersampling e oversampling (SMOTE)
    - Random Forest Classifier (com `class_weight='balanced'`)
    - Pipeline (`StandardScaler`+`LogisticRegression`) com ajuste de **threshold** de decisão
    - XGBoost (`scale_pos_weight` para lidar com o desbalanceamento)
    - Ajuste de hiperparâmetros via `GridSearchCV` (otimizando para `recall`)
5. Explicabilidade 
    - Importância das variáveis (`feature_importances_ do XGBoost)
    - **SHAP** - explica a contribuição de cada variável nas decisões do modelo

## Resultados (classe 1 - fraude)

| **Modelo** | **Precision** | **Recall** | **F1-score** |
|---|---|---|---|
| Regressão Logística (baseline) | 0.84 | 0.66 | 0.73 |
| Regressão Logística (threshold = 0.3) | 0.79 | 0.71 | 0.75 |
| Random Forest (`class_weight='balanced'`) | 0.84 | 0.76 | 0.79 |
| **XGBoost** | **0.94** | **0.78** | **0.85** |

- **AUC (ROC) da Regressão Logística**: ~0.93
- Melhor combinação encontrada via `GridSearchCV`para o XGBoost: `max_depth=10`, `n_estimators=200`

> Como o dataset é altamente desbalanceado, a **acurácia** não é uma métrica confiável — o foco da análise está em **recall** (capacidade de identificar fraudes reais) e **precision** (confiabilidade das fraudes apontadas), avaliados em conjunto pelo **F1-score** e pela **curva Precision-Recall**.

## Tecnologias e bibliotecas

- [Python](https://www.python.org/)
- [pandas](https://pandas.pydata.org/) — manipulação e análise de dados
- [NumPy](https://numpy.org/) — operações numéricas
- [scikit-learn](https://scikit-learn.org/) — pré-processamento, modelos (Logistic Regression, Random Forest), métricas e pipeline
- [imbalanced-learn](https://imbalanced-learn.org/) — balanceamento de classes (SMOTE)
- [XGBoost](https://xgboost.readthedocs.io/) — modelo de gradient boosting
- [SHAP](https://shap.readthedocs.io/) — explicabilidade do modelo
- [Matplotlib](https://matplotlib.org/) — visualização de dados

## Como executar

O projeto está em um único notebook Jupyter/Google Colab: `dio_deteccao_de_fraudes_bancarias.ipynb`.

1. Abra o notebook no [Google Colab](https://colab.research.google.com/) ou localmente com Jupyter.
2. Instale as dependências, caso necessário:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
   ```
3. Execute as células em ordem — o dataset é carregado automaticamente via URL, sem necessidade de download manual.

---

Projeto desenvolvido por [querycat](https://github.com/querycat).

# Projeto de Machine Learning: Previsão de Risco de Atrasos na Entrega (Olist Store)

Este projeto implementa um pipeline completo de Machine Learning comparando **Regressão Logística** e **K-Nearest Neighbors (KNN)** para predição antecipada de atrasos logísticos na Olist Store, priorizando a intervenção proativa e a retenção de clientes.

## Resultados dos Modelos

| Métrica | Regressão Logística (Modelo A) | KNN k=21 (Modelo B) |
| :--- | :---: | :---: |
| Recall (Classe 1) | 83,83% | 82,11% |
| Precision (Classe 1) | 55,92% | 60,16% |
| Accuracy (Exatidão) | 78,64% | 81,25% |
| F1-Score (Classe 1) | 67,09% | 69,44% |
| ROC AUC | 0,663 | 0,662 |

---

## Estrutura de Arquivos

- Projeto_Finalfazendo.docx: Relatório acadêmico completo e diagramado segundo as normas e especificações do projeto.
- [Projeto_Final_Olist_ML.ipynb](Projeto_Final_Olist_ML.ipynb): Jupyter Notebook executável com todo o fluxo de ponta a ponta (EDA, feature engineering, treino e gráficos).
- [pipeline_ml.py](pipeline_ml.py): Script Python autônomo e modular para execução direta no terminal.
- `outputs/`: Gráficos em alta resolução das matrizes de confusão comparativas, curvas ROC e distribuição de classes.

---

## Como Executar

### 1. Pré-requisitos

```bash
pip install pandas scikit-learn seaborn matplotlib python-docx openpyxl kagglehub
```

### 2. Execução do Script Python

```bash
python3 pipeline_ml.py
```

### 3. Execução do Jupyter Notebook

```bash
jupyter notebook Projeto_Final_Olist_ML.ipynb
```

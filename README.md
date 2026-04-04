# NLP CA3 — Abstract Classification Using ML and GPT Fine-Tuning

## Submitted By
- Shoeb Shakil Sutar (25070149022)
- Aryan Rajiender Sharma (25070149035)
- Symbiosis Institute of Technology, Pune

## What This Project Does

Classifies research paper abstracts into two categories using the UCI Inflation Abstracts dataset (https://doi.org/10.24432/C5ZC9S):
- 1 = related to inflation/economics/finance
- 0 = not related

We ran 8 model configurations total and compared their test set numbers.

## Models Tested

Approach A — sklearn classifiers with two vectorizers:
- LR, MNB, RF, LinearSVC × {CountVec, TF-IDF}

Approach B — OpenAI fine-tuning:
- GPT-4.1-nano via the fine-tuning API (no local GPU needed)

## Quick Numbers

| Setup | Acc. | Wt. F1 |
|---|---|---|
| LR + CV | 0.9078 | 0.91 |
| MNB + CV | 0.8728 | 0.88 |
| RF + CV | 0.8859 | 0.87 |
| SVC + CV | 0.8728 | 0.88 |
| LR + TFIDF | 0.8464 | 0.81 |
| RF + TFIDF | 0.9035 | 0.89 |
| SVC + TFIDF | **0.9210** | 0.89 |
| GPT-4.1-nano | 0.9015 | **0.90** |

## Repo Layout

- `notebooks/Traditional_ML_Implementation_Abstract.ipynb` — all sklearn models
- `notebooks/NLP_Fine_Tuning.ipynb` — GPT fine-tuning pipeline
- `requirements.txt` — dependencies
## Running the Code

sklearn notebook: Open in Colab → upload classified_abstracts.csv → run all.

Fine-tuning notebook: Open in Colab → store your OpenAI key in Colab Secrets as NLPCaseStudy → upload train/val/test CSVs → run all. API key is never hardcoded.

## Packages

Python 3.10+, scikit-learn, nltk, pandas, numpy, matplotlib, seaborn, openai

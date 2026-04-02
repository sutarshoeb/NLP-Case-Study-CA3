# A Comparative Analysis of Traditional Machine Learning and Fine-Tuned GPT-4.1-nano for Binary Text Classification of Research Paper Abstracts

## NLP Case Study — CA3 Submission

### Authors
- **Shoeb Shakil Sutar (25070149022)**
- **Aryan Rajiender Sharma (25070149035)**

---

### Overview

This project presents a comparative analysis of text classification methods applied to research paper abstracts. The task is binary classification — determining whether an abstract is relevant to inflation-related economics and finance topics (class 1) or not (class 0).

Two approaches are explored:
1. **Traditional ML Models** — Logistic Regression, Multinomial Naive Bayes, Random Forest, and Linear SVC with CountVectorizer and TF-IDF vectorization.
2. **Fine-Tuned LLM** — Supervised fine-tuning of OpenAI's GPT-4.1-nano model via the OpenAI fine-tuning API.

---

### Dataset

- **Source:** [UCI Machine Learning Repository — Inflation Research Abstracts Classification](https://doi.org/10.24432/C5ZC9S)
- **Records:** 1,138 (1,136 after deduplication)
- **Features:** DOI, Abstract, Label (binary)
- **Class Distribution:** ~79% class 0, ~21% class 1

---

### Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Traditional_ML_Implementation_Abstract.ipynb
│   └── NLP_Fine_Tuning.ipynb
└── figures/
    └── (generated during notebook execution)
```

---

### Results Summary

| Configuration | Test Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression + CountVec | 0.9078 | 0.91 |
| Multinomial Naive Bayes + CountVec | 0.8728 | 0.88 |
| Random Forest + CountVec | 0.8859 | 0.87 |
| Linear SVC + CountVec | 0.8728 | 0.88 |
| Logistic Regression + TF-IDF | 0.8464 | 0.81 |
| Random Forest + TF-IDF | 0.9035 | 0.89 |
| Linear SVC + TF-IDF | **0.9210** | 0.89 |
| GPT-4.1-nano (Fine-Tuned) | 0.9015 | **0.90** |

---

### How to Run

#### Traditional ML Notebook
1. Open `notebooks/Traditional_ML_Implementation_Abstract.ipynb` in Google Colab.
2. Upload the dataset CSV (`classified_abstracts.csv`) to the Colab runtime.
3. Run all cells sequentially.

#### Fine-Tuning Notebook
1. Open `notebooks/NLP_Fine_Tuning.ipynb` in Google Colab.
2. Add your OpenAI API key to Colab Secrets with the key name `NLPCaseStudy`.
3. Upload the train, validation, and test CSV splits to the Colab runtime.
4. Run all cells sequentially.

> **Note:** The fine-tuning notebook uses Google Colab's `userdata` secrets manager for API key handling. No API keys are hardcoded in the repository.

---

### Dependencies

See `requirements.txt` for the full list. Key libraries:
- Python 3.10+
- scikit-learn
- nltk
- pandas, numpy, matplotlib, seaborn
- openai (for fine-tuning notebook)

---

### References

1. A. Vaswani et al., "Attention is all you need," NeurIPS, 2017.
2. A. Radford et al., "Improving language understanding by generative pre-training," OpenAI, 2018.
3. C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, 1995.
4. L. Breiman, "Random forests," Machine Learning, 2001.
5. A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," AAAI Workshop, 1998.
6. C. D. Manning et al., Introduction to Information Retrieval, Cambridge University Press, 2008.
7. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, 2011.
8. S. Bird et al., Natural Language Processing with Python, O'Reilly Media, 2009.
9. T. Joachims, "Text categorization with support vector machines," ECML, 1998.
10. OpenAI, "Supervised fine-tuning," 2025. https://platform.openai.com/docs/guides/supervised-fine-tuning
11. D. A. Gonzalez, "Inflation research abstracts classification," UCI ML Repository, 2025. https://doi.org/10.24432/C5ZC9S

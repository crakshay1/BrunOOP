# Learn OOP the Hard Way

**Modeling and Implementing a Machine Learning Library**  
*Object-Oriented Programming with Python — Université Paris-Saclay*

###### By Akshay KHOKAN[@crakshay1](https://github.com/crakshay1), Georgy ZAOUK[@georgyzaouk](https://github.com/georgyzaouk) & Bruno YOUNG DE CASTRO[@zyrok12](https://github.com/zyrok12)

-------
⚠️ [View our project report.](https://github.com/MachineLearning-oop/2526-m1geniomhe-group6/blob/main/docs/report.pdf)

This project focuses on the design and implementation of a small machine learning library in Python, with an emphasis on clean, modular, and extensible software architecture. 

Inspired by existing machine learning libraries such as scikit-learn, the objective is to model core components including estimators, datasets, transformers, and pipelines without relying on external machine learning frameworks.

-----------------------------------------------------
## Repository structure

```
.
├── docs/
│   ├── class_diagrams/
│   │   ├── class_diagram_v1.png        # First version of the UML class diagram
│   │   ├── class_diagram_v2.png        # Intermediate version of the UML class diagram
│   │   ├── class_diagram_v3.png        # Intermediate version of the UML class diagram
│   │   └── ClassDiagram_VFINAL.png     # Final UML class diagram used in the report
│   │
│   └── report.pdf                      # Project report (PDF)                  
|
├── upsaclay-ml/
│   ├── upsaclay_ml/
│   │   ├── models/
│   │   │   ├── estimator.py            # Base estimator abstraction
│   │   │   ├── linear_model.py         # Linear models
│   │   │   ├── tree.py                 # Tree-based models
│   │   │   └── cluster.py              # Clustering models
│   │   │
│   │   ├── datasets.py                 # Dataset loading utilities
│   │   ├── transformers.py             # Data preprocessing transformers
│   │   ├── pipeline.py                 # Pipeline implementation
│   │   ├── model_selection.py          # Train/test split utilities
│   │   ├── metrics.py                  # Evaluation metrics
│   │   └── test_dataset.csv            # Example dataset
│   │
│   ├── test_calling_pipeline.py        # Pipeline tests
│   ├── test_supervised_learning.py     # Supervised learning tests
│   ├── test_unsupervised_learning.py   # Unsupervised learning tests
│   ├── test_MNIST_TCGA.py              # MNIST and TCGA dataset tests
│   ├── test_BrunOOP.py                 # OOP behavior tests
│   └── test_BrunOOP_SK.py              # Comparison with scikit-learn
│
├── 2526-m1geniomhe-OOPwithPython-Project.pdf      # Project instructions (PDF)
├── requirements.txt                               # Project dependencies
└── README.md                                      # Project documentation
```
-------------------------------------------------------------------------------------

## How to use

1. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the test scripts**

The library can be tested using the provided `test_*.py` scripts located in the `upsaclay-ml/` directory.

```bash
cd upsaclay-ml
python test_calling_pipeline.py
```

Other test scripts can be run in the same way to test different parts of the library.

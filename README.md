# Cognitive Load & Student Engagement Detection

A multimodal system that detects how cognitively loaded and engaged a student is during online learning - using eye-tracking metrices and facial expressions and recommends teaching interventions.

---

## How It Works

Two separate ML models run in parallel and their outputs are combined:

- **Cognitive Load model** - Logistic Regression on eye-tracking features (pupil dilation, blink rate, fixation duration, saccade velocity)
- **Student Engagement model** - XGBoost on HOG features extracted from facial images
- **Fusion layer** - combines both outputs into one of 9 learner states, then triggers a recommended intervention

---

## Repo Structure

```
├── Cognitive_Load/
│   ├── CL_EDA_Preprocessing.ipynb
│   └── CL_Model_Training.ipynb
├── Student_Engagement/
│   ├── se_eda_preprocessing.ipynb
│   └── se_model_training.ipynb
├── fusion_layer.ipynb
└── README.md
```

---

## Results

| Component | Macro F1 |
|---|---|
| Cognitive Load model | 0.881 (ROC-AUC = 0.974) |
| Student Engagement model | 0.901 |
| Fused system | 0.892 |

---

## Setup

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn opencv-python jupyterlab
```

Run the notebooks in order: CL pipeline → SE pipeline → fusion layer.

---

## Datasets

Both datasets are publicly available on Kaggle - anonymised eye-tracking records (n=3,000) and facial images (n=2,085).

---

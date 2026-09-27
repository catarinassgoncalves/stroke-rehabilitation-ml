# Machine Learning for Stroke Rehabilitation

Machine learning project focused on analyzing real-world physical therapy sessions from **stroke survivors**, using body-pose data extracted with MediaPipe.

The project explores three complementary tasks: non-linear sensor regression, physical therapy exercise recognition, and identification of the side of the body affected by stroke. Particular attention is given to **patient-aware validation, biomechanical preprocessing, feature engineering, and model selection** to improve generalization across unseen patients.

## Project Overview

The project is divided into three machine learning tasks:

1. **Non-linear Sensor Regression** — estimate measurements from a high-precision sensor using six lower-cost sensors.
2. **Exercise Classification** — distinguish between physical therapy exercises performed by stroke survivors.
3. **Impaired-Side Classification** — identify whether a patient's left or right side is affected using temporal skeleton sequences.

The classification tasks use pose information extracted from physical therapy sessions involving stroke survivors, represented through **33 body keypoints obtained with MediaPipe**.

---

## 1. Non-linear Sensor Regression

The first task investigates whether measurements from an expensive high-precision sensor can be estimated using six lower-cost sensors.

Several regression approaches were compared:

- Polynomial Regression with Ridge regularization
- Polynomial Regression with Lasso regularization
- Polynomial Kernel Ridge Regression
- RBF Kernel Ridge Regression
- Polynomial Support Vector Regression
- RBF Support Vector Regression

### Results

| Model | Validation R² | Validation MSE |
| --- | ---: | ---: |
| Polynomial Regression + Ridge | 0.9398 | 0.1371 |
| Polynomial Regression + Lasso | 0.8154 | 0.4203 |
| Polynomial Kernel Ridge | 0.9451 | 0.1250 |
| **RBF Kernel Ridge** | **0.9849** | **0.0343** |
| Polynomial SVR | 0.9797 | 0.0463 |
| RBF SVR | 0.9848 | 0.0345 |

**RBF Kernel Ridge Regression** achieved the best validation performance and was selected as the final model.

---

## 2. Physical Therapy Exercise Classification

The second task focuses on recognizing physical therapy exercises from body-pose features.

The data contains pose information from **18 stroke survivors**, represented through the mean and standard deviation of the positions of 33 body keypoints.

The objective is to distinguish between three exercises:

- **E1** — brushing hair
- **E2** — brushing teeth
- **E5** — hip flexion

### Preprocessing and Feature Engineering

Several transformations were explored to reduce inter-patient variability and better represent the underlying movement:

- **Body normalization** based on torso height and shoulder width.
- **Movement mirroring**, representing movements consistently regardless of whether they were performed on the left or right side.
- **Active-side information**, indicating which side of the body exhibits greater movement.
- Exploration of geometric features based on body-joint positions, distances, and angles.

These transformations improved the ability of the models to distinguish exercises with similar movement patterns, particularly E1 and E2.

### Patient-Aware Validation

A random train-validation split could place observations from the same patient in both sets, producing an overly optimistic estimate of generalization.

To avoid this, model evaluation was performed using **GroupKFold cross-validation grouped by patient**.

For the main experiments, seven folds were used:

```text
12 patients → training
 2 patients → validation
```

This ensures that the model is evaluated on patients that were not used to train it.

### Model Comparison

| Model | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Random Forest | 0.822 | 0.800 |
| Logistic Regression | 0.851 | 0.833 |
| **RBF SVM** | **0.917** | **0.903** |
| MLP | 0.824 | 0.807 |

The **RBF Support Vector Machine** achieved the strongest performance.

Feature engineering also improved the SVM Macro F1 score from approximately **0.887 to 0.917**.

---

## 3. Impaired-Side Classification

The final task addresses a more complex problem: identifying whether the patient's **left or right side** is affected by stroke.

Unlike the previous task, each observation contains a **temporal sequence of skeleton poses** rather than precomputed summary statistics.

### Sequence Preprocessing

Physical therapy sessions have different durations and therefore produce sequences with different numbers of frames.

To obtain a fixed-dimensional representation, the skeleton sequences were:

1. normalized according to body proportions;
2. temporally aligned using **cubic spline interpolation**;
3. reduced to exercise-specific sets of relevant body joints.

A larger feature-engineering approach was also explored using features such as:

- joint distances;
- angles;
- velocities;
- moving averages;
- movement asymmetries;
- ranges of motion;
- percentiles;
- path lengths.

The experiments showed that the simpler representation based on selected joint positions generalized better than the larger engineered feature set.

### Exercise-Specific Models

Different exercises emphasize different body regions and movement patterns. Instead of training a single global classifier, separate models were trained for different exercises:

```text
              ┌── E1 + E2 classifier
              │
Patient ──────├── E3 classifier
              │
              ├── E4 classifier
              │
              └── E5 classifier
                       │
                       ▼
               Weighted Soft Voting
                       │
                       ▼
               Impaired Side
```

E1 and E2 were combined due to their similar movement characteristics and smaller number of observations.

The probabilities produced by the four classifiers are aggregated at the patient level and combined through **weighted soft voting**.

### Patient-Level Cross-Validation

Evaluation was again performed at the patient level.

A **14-fold GroupKFold** strategy was used, effectively evaluating the models on one held-out patient at a time:

```text
13 patients → training
 1 patient  → validation
```

This provides a more realistic estimate of performance on unseen patients.

### Model Comparison

Balanced accuracy obtained during patient-aware cross-validation:

| Model | E1 + E2 | E3 | E4 | E5 | Ensemble |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.286 | 0.500 | 0.286 | 0.571 | 0.571 |
| Logistic Regression | 0.500 | 0.571 | 0.500 | 0.571 | 0.500 |
| CatBoost | 0.714 | 0.571 | 0.429 | 0.500 | 0.714 |
| XGBoost | 0.643 | 0.714 | 0.643 | 0.500 | 0.571 |
| **RBF SVM** | **0.929** | **0.714** | **0.857** | **0.929** | **0.929** |

The final approach uses **RBF SVM classifiers with weighted soft voting**, achieving a cross-validation balanced accuracy of **0.929**.

---

## Key Takeaways

- **Patient-aware validation matters:** grouping observations by patient prevents information leakage and provides a more realistic estimate of performance on unseen subjects.
- **Domain-specific preprocessing improved classification:** body normalization, movement mirroring, and active-side information increased exercise-classification performance.
- **More features did not necessarily improve generalization:** a relatively compact representation based on selected body joints performed better than a much larger set of engineered temporal and geometric features.
- **Exercise-specific modeling was effective:** combining specialized classifiers through weighted soft voting achieved strong patient-level cross-validation performance.

---

## Repository Structure

```text
stroke-rehabilitation-ml/
├── notebooks/
│   ├── sensor_regression.ipynb
│   ├── exercise_classification.ipynb
│   └── impaired_side_classification.ipynb
│
├── src/
│   ├── regression/
│   │   └── model.py
│   │
│   ├── exercise_classification/
│   │   ├── model.py
│   │   └── preprocessing.py
│   │
│   └── impaired_side_classification/
│       ├── model.py
│       └── preprocessing.py
│
├── models/
│   ├── regression_model.pkl
│   ├── exercise_classifier.pkl
│   └── impaired_side/
│       ├── model_E12.pkl
│       ├── model_E3.pkl
│       ├── model_E4.pkl
│       └── model_E5.pkl
│
├── report/
│   └── project_report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/catarinassgoncalves/stroke-rehabilitation-ml.git
cd stroke-rehabilitation-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

The datasets used in the original project are **not included in this repository**.

To reproduce the notebook experiments, the expected local structure is:

```text
data/
├── X_train.npy
├── Y_train.npy
├── Xtrain1.pkl
├── Ytrain1.npy
├── Xtrain2.pkl
└── Ytrain2.npy
```

The `data/` directory is excluded from version control.

---

## Technologies

- **Python**
- **NumPy**
- **Pandas**
- **SciPy**
- **scikit-learn**
- **Optuna**
- **XGBoost**
- **CatBoost**
- **Matplotlib**
- **Seaborn**
- **MediaPipe pose data**

---

## Project Context

This project was developed as part of the **Machine Learning** course at **Instituto Superior Técnico (University of Lisbon)**.

It was completed collaboratively by **Catarina Gonçalves and André Amaro**.

The repository contains the experimental notebooks, preprocessing pipelines, and trained models used throughout the project.

For a complete description of the methodology, experiments, and analysis, see the [project report](report/project_report.pdf).
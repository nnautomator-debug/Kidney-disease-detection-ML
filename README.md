# 🩺 Chronic Kidney Disease (CKD) Clinical Screening Pipeline

A leak-free, production-grade Machine Learning pipeline engineered to classify Chronic Kidney Disease (CKD) from routine lab test results and clinical observations. Built using Scikit-Learn `Pipeline` and `ColumnTransformer` architectures, this project prioritizes clinical safety by optimizing for high sensitivity (recall) to minimize dangerous false-negative diagnoses.

---

## 📌 Executive Summary & Key Results

- **Dataset**: UCI Chronic Kidney Disease Dataset (400 patient records, 25 clinical features).
- **Validation**: 5-Fold Stratified Cross-Validation (Strict train/test isolation inside cross-validation loops).
- **Primary Metric**: **98.00% ± 1.70%** Mean Accuracy across folds.
- **Clinical Safety**: **$\ge$ 98.0% Sensitivity (Recall)** maintained across all folds, ensuring minimal missed CKD cases.
- **Architectural Highlight**: Zero data leakage via encapsulated imputation, scaling, and categorical encoding within Scikit-Learn execution pipelines.

---

## 🏥 Clinical Context & Metric Prioritization

In diagnostic medical screening, **Type I errors (False Positives)** and **Type II errors (False Negatives)** carry asymmetrical risks:

- ❌ **False Positive (Type I)**: A healthy patient is flagged for CKD $\rightarrow$ *Consequence*: Follow-up blood/urine test ordered. Minor cost/inconvenience.
- 🚨 **False Negative (Type II)**: A CKD patient is classified as healthy $\rightarrow$ *Consequence*: Disease progresses silently to irreversible renal failure. **Catastrophic.**

**Engineering Objective**: Design a classifier that maximizes **Sensitivity (Recall)** for the `ckd` class while maintaining high overall precision, establishing a safe initial triage mechanism for general practitioners.

---

## 📊 Dataset Architecture & Feature Engineering

The pipeline ingests 25 raw clinical features spanning laboratory measurements, physical symptoms, and patient history:

| Category | Features Included |
| :--- | :--- |
| **Vital Signs & Demographics** | `age`, `bp` (Blood Pressure) |
| **Urinalysis (Lab)** | `sg` (Specific Gravity), `al` (Albumin), `su` (Sugar), `rbc` (Red Blood Cells), `pc` (Pus Cell), `pcc` (Pus Cell Clumps), `ba` (Bacteria) |
| **Blood Chemistry (Lab)** | `bgr` (Blood Glucose Random), `bu` (Blood Urea), `sc` (Serum Creatinine), `sod` (Sodium), `pot` (Potassium), `hemo` (Hemoglobin), `pcv` (Packed Cell Volume), `wc` (WBC Count), `rc` (RBC Count) |
| **Medical History & Symptoms** | `htn` (Hypertension), `dm` (Diabetes Mellitus), `cad` (Coronary Artery Disease), `appet` (Appetite), `pe` (Pedal Edema), `ane` (Anemia) |
| **Target Variable** | `classification` (`1` = `ckd`, `0` = `notckd`) |

---

## 🛡️ Leak-Free Pipeline Architecture

A major flaw in many medical ML implementations is **data leakage** caused by imputing missing values or scaling features across the entire dataset prior to splitting. 

To eliminate leakage:
1. Data cleaning normalizes raw text encodings and maps missing tokens (`?`, `\t`, blank strings) to `np.nan`.
2. Imputation statistics (median for numerical, most frequent for categorical) and scaling params are **fit exclusively on training folds** using `ColumnTransformer`.
3. Test sets remain completely unseen until inference.

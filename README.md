# 🧠 Multimodal MRI Biomarker Analysis (T1 + T2*)

## 📌 Overview

This project explores the use of **multimodal MRI data** to identify early biomarkers of cognitive decline.

It integrates:

* **Structural MRI (T1)** → brain atrophy
* **Susceptibility-sensitive MRI (T2*)** → iron / microstructural changes

---

## 🧠 Research Context

This project was developed as part of preparation for PhD-level research in medical imaging and AI.

Two approaches were implemented:

### 1️⃣ T1-Only Model

* Structural feature extraction
* Hippocampus proxy
* Atrophy indicators

### 2️⃣ Multimodal Model (T1 + T2*)

* Deep grey matter features (basal ganglia, thalamus)
* Hemispheric asymmetry
* Combined structural + microstructural modelling

---

## 🔍 Key Findings

* Structural MRI provides baseline predictive features
* T2* features (deep brain + asymmetry) add complementary information
* Multimodal approach captures both **atrophy and microstructural changes**

⚠️ Results are exploratory due to small dataset

---

## 🧪 Methodology

MRI → Preprocessing → Feature Extraction → Machine Learning → Interpretation

---

## ⚠️ Limitations

* Small sample size
* Approximate ROI selection
* Simulated labels

---

## 🚀 Future Work

* Atlas-based segmentation
* Multi-echo T2* mapping
* Clinical data integration
* Explainable AI (SHAP)

---

## ▶️ Run

```bash
python analysis/multimodal_pipeline.py
```

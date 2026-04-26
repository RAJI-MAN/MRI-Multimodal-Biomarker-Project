import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ----------------------------
# DATA PATH
# ----------------------------
base_dir = "data"

features = []
labels = []

# ----------------------------
# LOAD MRI
# ----------------------------
for subject in os.listdir(base_dir):

    subject_path = os.path.join(base_dir, subject)

    if not os.path.isdir(subject_path):
        continue

    files = os.listdir(subject_path)
    nii_files = [f for f in files if f.endswith(".nii") or f.endswith(".nii.gz")]

    if len(nii_files) == 0:
        continue

    mri_path = os.path.join(subject_path, nii_files[0])
    print(f"Loading: {mri_path}")

    img = nib.load(mri_path)
    data = img.get_fdata()

    # FEATURES
    global_mean = np.mean(data)
    global_std = np.std(data)

    mid_slice = data[:, :, data.shape[2] // 2]
    slice_mean = np.mean(mid_slice)

    # Hippocampus proxy
    z = int(data.shape[2] * 0.4)
    x1, x2 = int(data.shape[0]*0.3), int(data.shape[0]*0.7)
    y1, y2 = int(data.shape[1]*0.3), int(data.shape[1]*0.7)

    hippocampus = data[x1:x2, y1:y2, z]
    hippocampus_mean = np.mean(hippocampus)

    # Atrophy
    csf_ratio = np.sum(data < 500) / data.size
    wm_ratio = np.sum(data > 1500) / data.size

    age = np.random.randint(50, 85)

    # ⚡ NEW BALANCED LABEL (BETTER)
    if (age > 75 and csf_ratio > 0.25) or hippocampus_mean < 850:
        label = 1
    else:
        label = 0

    features.append([
        global_mean,
        global_std,
        slice_mean,
        hippocampus_mean,
        csf_ratio,
        wm_ratio,
        age
    ])

    labels.append(label)

# ----------------------------
# DATAFRAME
# ----------------------------
columns = [
    "global_mean", "global_std", "slice_mean",
    "hippocampus_mean", "csf_ratio", "wm_ratio", "age"
]

df = pd.DataFrame(features, columns=columns)
df["label"] = labels

print("\nDataset:\n", df)
print("\nLabel Distribution:\n", df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# ----------------------------
# MODEL
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X, y)

# ----------------------------
# SAFE CROSS-VALIDATION
# ----------------------------
if min(df["label"].value_counts()) >= 2:
    scores = cross_val_score(rf_model, X, y, cv=2)
    print("\nCross-validation scores:", scores)
    print("Mean CV score:", scores.mean())
else:
    print("\n⚠️ Skipping cross-validation (not enough samples per class)")

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
importance = rf_model.feature_importances_

print("\nFeature Importance:\n", dict(zip(X.columns, importance)))

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(8,4))
plt.bar(X.columns, importance)
plt.title("Final Model Biomarkers")
plt.xticks(rotation=30)

os.makedirs("results", exist_ok=True)
plt.savefig("results/final_model.png")

plt.show()
# ----------------------------
# SAVE TERMINAL RESULTS
# ----------------------------
os.makedirs("results", exist_ok=True)

# Save dataset
df.to_csv("results/t1_dataset.csv", index=False)

# Save feature importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
})
importance_df.to_csv("results/t1_feature_importance.csv", index=False)

# Save terminal-style output
with open("results/t1_output.txt", "w") as f:

    f.write("===== DATASET =====\n")
    f.write(df.to_string())
    f.write("\n\n")

    f.write("===== LABEL DISTRIBUTION =====\n")
    f.write(str(df["label"].value_counts()))
    f.write("\n\n")

    # Save CV results (if available)
    if min(df["label"].value_counts()) >= 2:
        f.write("===== CROSS VALIDATION =====\n")
        f.write(f"Scores: {scores}\n")
        f.write(f"Mean: {scores.mean()}\n\n")
    else:
        f.write("===== CROSS VALIDATION =====\n")
        f.write("Skipped (not enough samples per class)\n\n")

    f.write("===== FEATURE IMPORTANCE =====\n")
    for name, val in zip(X.columns, importance):
        f.write(f"{name}: {val:.4f}\n")

print("\n✅ T1 results saved in 'results/' folder")
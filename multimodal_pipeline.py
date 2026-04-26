import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# PATH
# ----------------------------
base_dir = "data"

features = []
labels = []

# ----------------------------
# LOOP THROUGH SUBJECTS
# ----------------------------
for subject in os.listdir(base_dir):

    subject_path = os.path.join(base_dir, subject)

    if not os.path.isdir(subject_path):
        continue

    t1_path = os.path.join(subject_path, "T1.nii")
    t2_path = os.path.join(subject_path, "T2star.nii")

    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        print(f"Skipping {subject} (missing files)")
        continue

    print(f"Processing: {subject}")

    # ----------------------------
    # LOAD IMAGES
    # ----------------------------
    t1 = nib.load(t1_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()

    # ----------------------------
    # NORMALISATION (CRITICAL)
    # ----------------------------
    t1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-8)
    t2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-8)

    # ----------------------------
    # T1 FEATURES (STRUCTURAL)
    # ----------------------------
    t1_mean = np.mean(t1)
    t1_std = np.std(t1)

    mid_slice = t1[:, :, t1.shape[2] // 2]
    t1_slice_mean = np.mean(mid_slice)

    # Hippocampus proxy
    z = int(t1.shape[2] * 0.4)
    x1, x2 = int(t1.shape[0]*0.3), int(t1.shape[0]*0.7)
    y1, y2 = int(t1.shape[1]*0.3), int(t1.shape[1]*0.7)

    hippo_t1 = t1[x1:x2, y1:y2, z]
    hippo_t1_mean = np.mean(hippo_t1)

    # Atrophy indicators
    csf_ratio = np.sum(t1 < -0.5) / t1.size
    wm_ratio = np.sum(t1 > 0.5) / t1.size

    # ----------------------------
    # T2* FEATURES (IMPROVED 🔥)
    # ----------------------------
    t2_mean = np.mean(t2)
    t2_std = np.std(t2)

    # Hippocampus (T2*)
    hippo_t2 = t2[x1:x2, y1:y2, z]
    hippo_t2_mean = np.mean(hippo_t2)

    # Dimensions
    x, y, z_dim = t2.shape

    # 1. Basal Ganglia (iron-rich)
    bg_region = t2[
        int(x*0.4):int(x*0.6),
        int(y*0.4):int(y*0.6),
        int(z_dim*0.5)
    ]
    bg_t2_mean = np.mean(bg_region)

    # 2. Thalamus
    thalamus_region = t2[
        int(x*0.35):int(x*0.65),
        int(y*0.35):int(y*0.65),
        int(z_dim*0.6)
    ]
    thalamus_t2_mean = np.mean(thalamus_region)

    # 3. Hemispheric asymmetry
    left_region = t2[:int(x/2), :, int(z_dim*0.5)]
    right_region = t2[int(x/2):, :, int(z_dim*0.5)]

    asymmetry_t2 = np.mean(left_region) - np.mean(right_region)

    # ----------------------------
    # SIMULATED CLINICAL DATA
    # ----------------------------
    age = np.random.randint(50, 85)

    # ----------------------------
    # LABEL (balanced logic)
    # ----------------------------
    if (age > 75 and csf_ratio > 0.2) or bg_t2_mean < -0.2:
        label = 1
    else:
        label = 0

    # ----------------------------
    # STORE FEATURES
    # ----------------------------
    features.append([
        t1_mean,
        t1_std,
        t1_slice_mean,
        hippo_t1_mean,
        csf_ratio,
        wm_ratio,
        t2_mean,
        t2_std,
        hippo_t2_mean,
        bg_t2_mean,
        thalamus_t2_mean,
        asymmetry_t2,
        age
    ])

    labels.append(label)

# ----------------------------
# DATAFRAME
# ----------------------------
columns = [
    "t1_mean",
    "t1_std",
    "t1_slice",
    "hippo_t1",
    "csf_ratio",
    "wm_ratio",
    "t2_mean",
    "t2_std",
    "hippo_t2",
    "bg_t2",
    "thalamus_t2",
    "asymmetry_t2",
    "age"
]

df = pd.DataFrame(features, columns=columns)
df["label"] = labels

print("\nDataset:\n", df)
print("\nLabel Distribution:\n", df["label"].value_counts())

# ----------------------------
# MODEL
# ----------------------------
X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
importance = model.feature_importances_

print("\nFeature Importance:\n")
for name, val in zip(X.columns, importance):
    print(f"{name}: {val:.4f}")

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(12,4))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Multimodal MRI Biomarkers (Improved T2* Regions)")

os.makedirs("results", exist_ok=True)
plt.savefig("results/improved_multimodal.png")

plt.show()
# ----------------------------
# SAVE RESULTS TO FILE
# ----------------------------
os.makedirs("results", exist_ok=True)

with open("results/output.txt", "w") as f:
    f.write("Dataset:\n")
    f.write(df.to_string())
    f.write("\n\nFeature Importance:\n")

    for name, val in zip(X.columns, importance):
        f.write(f"{name}: {val:.4f}\n")
import glob
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

ROOT = "/media/sf_flyvideos/fly_videos_00_00_100_deepethogram/DATA"

#get all prediction files
pred_files = sorted(glob.glob(f"{ROOT}/*/*_predictions.csv"))

y_true_all = []
y_pred_all = []

for pred_path in pred_files:
    # find matching ground truth
    gt_path = pred_path.replace("_predictions.csv", "_ground_truth.csv")

    if not os.path.exists(gt_path):
        print(f"Skipping (no GT): {pred_path}")
        continue

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # match lengths safely
    min_len = min(len(pred_df), len(gt_df))

    y_pred_all.append(pred_df["asleep"][:min_len])
    y_true_all.append(gt_df["asleep"][:min_len])

#combine everything
y_pred = pd.concat(y_pred_all, ignore_index=True)
y_true = pd.concat(y_true_all, ignore_index=True)

print("Final lengths:", len(y_true), len(y_pred))

#confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=[0, 1]   # ensures both classes appear
)

#save
fig = disp.figure_
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=300)
plt.close(fig)

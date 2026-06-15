import pandas as pd
import numpy as np
import glob
import os.path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Get all prediction and ground truth files
DATA_FOLDER = "/media/sf_flyvideos/fly_videos_00_00_100_deepethogram/DATA/"

# 2. Store all data
all_data = []

# 3. Loop through all files
chunks = range(100, 120)

for chunk in chunks:

    pred_path = os.path.join(
        DATA_FOLDER,
        f"FlyHostel3_2X_2025-12-19_17-00-00_{str(chunk).zfill(6)}_001",
        f"{str(chunk).zfill(6)}_predictions.csv"
    )

    gt_path = os.path.join(
        DATA_FOLDER,
        f"FlyHostel3_2X_2025-12-19_17-00-00_{str(chunk).zfill(6)}_001",
        f"{str(chunk).zfill(6)}_ground_truth.csv"
    )

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    all_df = pred_df.copy()
    del all_df["moving"]

    all_df["chunk"] = chunk
    nrows = all_df.shape[0]
    all_df["frame_idx"] = np.arange(nrows)
    # asleep, chunk, frame_idx
    all_df.rename({"asleep": "predictions"}, axis=1, inplace=True)
    # predictions, chunk, frame_idx

    all_df["ground_truth"] = gt_df["asleep"]
    # predictions, chunk, frame_idx, ground_truth

    all_data.append(all_df)

# 4. Combine everything
# all_data is a list
all_data = pd.concat(all_data, ignore_index=True)
# all_data is a dataframe

# 5. DEBUG: check class distribution
print("y_true distribution:\n", all_data["ground_truth"].value_counts())
print("y_pred distribution:\n", all_data["predictions"].value_counts())


# find frames where predictions says the fly is awake but ground truth says it is awake
false_negatives = all_data.loc[
    (all_data["predictions"] == False) & (all_data["ground_truth"] == True)
]
false_positives = all_data.loc[
    (all_data["predictions"] == True) & (all_data["ground_truth"] == False)
]

false_negatives.to_csv("false_negatives.csv")
false_positives.to_csv("false_positives.csv")

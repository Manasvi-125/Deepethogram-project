import os.path

### CHANGE THIS TO WHEREVER DATA/ IS IN YOUR COMPUTER!!!! 
output_folder="C:/Users/Manasvi/Desktop/flyvideos/fly_videos_00_00_100_deepethogram/DATA/"
assert os.path.exists(output_folder), \
    f"Please make sure output_folder poins to your DATA/ folder"

# sleep_utils.py
import glob
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
time_window_length=1 # this means 1 point every 1 second

def rle(x):
    """
    Find runs of consecutive items in an array

    params:
    @x = ID numpy array  
    
    returns three arrays containg the run values, the start indices of the runs, and the lengths of the runs 
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
        
        
def sleep_contiguous(moving, fs, min_valid_time = 300):
    """ 
    Checks if contiguous bouts of immobility are greater than the minimum valid time given

    Params:
    @moving = pandas series, series object comtaining the movement data of individual flies
    @fs = int, sampling frequency (Hz) to scale minimum length to time in seconds
    @min_valid_time = min amount immobile time that counts as sleep, default is 300 (i.e 5 mins) 
    
    returns a list object to be added to a pandas dataframe
    """
    min_len = fs * min_valid_time
    r_sleep =  rle(np.logical_not(moving))
    valid_runs = r_sleep[2] >= min_len 
    r_sleep_mod = valid_runs & r_sleep[0]
    r_small = []
    for c, i in enumerate(r_sleep_mod):
        r_small += ([i] * r_sleep[2][c])

    return r_small
    
def sleep_annotation(df):
    df["asleep"]=sleep_contiguous(
        df["active_state"],
        1/time_window_length,
        min_valid_time=300 # 300 seconds = 5 minutes
    )
    return df
 
 
# main.py
experiment="FlyHostel3_2X_2025-12-19_17-00-00"
labels_suffix="*labels.csv"

run_qc=False

index=pd.read_csv(f"{experiment}_index.csv", index_col=0)

# read labels
folders=glob.glob(os.path.join(output_folder, f"{experiment}_*"))
data=[]
for folder in folders:
    labels_file=glob.glob(os.path.join(folder, labels_suffix))
    if labels_file:
        labels_file=labels_file[0]
    else:
        labels_file=None
        continue

    key = os.path.basename(folder.rstrip(os.path.sep))
    labels=pd.read_csv(labels_file, index_col=0)
    labels["frame_idx"]=np.arange(labels.shape[0])

    labels["key"]=key
    data.append(labels)

data=pd.concat(data, axis=0).reset_index(drop=True)
data=data.merge(index, on="key", how="left")


# annotate: is the fly in an activate state, or inactive state?
active_behaviors=["walk", "background", "groom", "feed"]
inactive_behaviors=["inactive"]
other_behaviors=["pe", "micromovement", "twitch"]

data["active_state"]=(data["background"]==1) | (data["walk"] == 1) | (data["groom"] == 1) | (data["feed"] == 1) | (
    (data[active_behaviors + inactive_behaviors + other_behaviors]==-1).all(axis=1)
)
data["inactive_state"]=(data["inactive"]==1)


# qc
problematic_rows=data.loc[(data["active_state"]==data["inactive_state"])]
problematic_rows=problematic_rows[["key", "frame_idx"]]
problematic_rows.to_csv("problematic_rows.csv")

if run_qc and problematic_rows.shape[0]>0:
    raise ValueError("Incompatible behaviors detected. Please correct frames written in problematic_rows.csv")


data_asleep=data.groupby("id").apply(sleep_annotation)[["key", "chunk", "active_state", "asleep"]].rename({"active_state": "moving"}, axis=1)

data_asleep["moving"]=data_asleep["moving"].astype(int)
data_asleep["asleep"]=data_asleep["asleep"].astype(int)

for key, gt_data in tqdm(data_asleep.groupby("key")):
    chunk=gt_data["chunk"].iloc[0]
    folder=f"{output_folder}/{key}"

    os.makedirs(folder, exist_ok=True)
    dest_file = f"{folder}/{str(chunk).zfill(6)}_ground_truth.csv"
    print(dest_file)
	  gt_data[["moving", "asleep"]].reset_index(drop=True).to_csv(f"{dest_file}")

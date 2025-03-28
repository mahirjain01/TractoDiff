import pickle
import os
import random
from os.path import join

DATA_ROOT = "data_sample"
DATA_PKL_PATH = join(DATA_ROOT, "data.pkl")

KEEP_PERCENT = 0.02

with open(DATA_PKL_PATH, "rb") as f:
    data = pickle.load(f)

file_list = data["file_names"] 
print(f"Original length of file_list: {len(file_list)}")

random.shuffle(file_list)
keep_count = int(len(file_list) * KEEP_PERCENT)
subsampled = file_list[:keep_count]
print(f"Keeping {keep_count} files out of {len(file_list)}")


data["file_names"] = subsampled

SAVE_PATH = join(DATA_ROOT, "data_subsample.pkl") 
with open(SAVE_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"Saved new data.pkl at: {SAVE_PATH}")

import pickle
import numpy as np
from typing import Any

def get_data_shape(value: Any) -> str:

    if isinstance(value, np.ndarray):
        return str(value.shape)
    elif isinstance(value, list):
        if len(value) == 0:
            return "empty list"
        try:
            return str(np.array(value).shape)
        except:
            return f"list of length {len(value)}"
    else:
        return str(type(value))

def main():

    file_path = '/tracto/TractoDiff/data_sample/data_folder/0_2.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Data shapes:")
    for key, value in data.items():
        print(f"{key}: {get_data_shape(value)}")
    
    # print(data['vel'])

if __name__ == "__main__":
    main()

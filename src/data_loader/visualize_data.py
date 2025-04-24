import pickle
import numpy as np
from typing import Any

def get_data_shape(value: Any) -> str:
    """
    Determines the shape or type description of a given value.

    Args:
        value: The data item (could be numpy array, list, or other type).

    Returns:
        A string describing the shape or type.
    """
    if isinstance(value, np.ndarray):
        # If it's a numpy array, return its shape as a string
        return str(value.shape)
    elif isinstance(value, list):
        # If it's a list
        if len(value) == 0:
            return "empty list"
        try:
            # Try converting to a numpy array to get its shape
            # This works well for lists of lists (or lists of numbers)
            return str(np.array(value).shape)
        except:
            # If conversion fails (e.g., list of mixed types), return length
            return f"list of length {len(value)}"
    else:
        # For any other type, return the type name
        return str(type(value))

def main():
    """
    Loads data from a pickle file, prints the shape/type of each item,
    and prints the first trajectory.
    """
    # Define the path to the pickle file
    # IMPORTANT: Ensure this path is correct for your system
    file_path = '/tracto/TractoDiff/data_sample/data_folder/0_2.pkl' # Example path

    try:
        # Open the file in binary read mode
        with open(file_path, 'rb') as f:
            # Load the data using pickle
            data = pickle.load(f)

        print("Data shapes:")
        # Iterate through the dictionary items (key-value pairs)
        for key, value in data.items():
            # Print the key and the shape/type description of the value
            print(f"{key}: {get_data_shape(value)}")

            if(key == 'targets'):
                print("The targets are: ", value[0])
            
            if(key == 'lidar'):
                print("The lidar is: ", value)

        # --- Print the first trajectory ---
        if 'trajectories' in data:
            trajectories_data = data['trajectories']
            if isinstance(trajectories_data, (np.ndarray, list)) and len(trajectories_data) > 0:
                print("\n--- First Trajectory ---")
                # Select the first trajectory (at index 0)
                first_trajectory = trajectories_data[0]
                print(first_trajectory)
                # You could add more specific printing/formatting here if needed
                # For example, print each point on a new line:
                # print("\nPoints in the first trajectory:")
                # for point in first_trajectory:
                #    print(point)
            else:
                print("\n'trajectories' key found, but it's empty or not a sequence.")
        else:
            print("\n'trajectories' key not found in the data.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle data from '{file_path}'. The file might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Standard Python entry point check
if __name__ == "__main__":
    main()

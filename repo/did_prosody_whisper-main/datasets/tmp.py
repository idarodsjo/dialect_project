import pickle
import pandas as pd

file_path = '/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/test_data.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        print("Beginning of loaded data:")
        print("Columns: ", data.columns if hasattr(data, 'columns') else 'N/A')
        if isinstance(data, list):
            print(data[:5])  # Print first 5 elements of a list
        elif isinstance(data, dict):
            # Print first few key-value pairs of a dictionary
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:
                    break
                print(f"{key}: {value}")
        elif isinstance(data, pd.DataFrame):
            print(data.head()) # Print first 5 rows of a DataFrame
        else:
            print(str(data)[:200]) # Print first 200 characters of other object types
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
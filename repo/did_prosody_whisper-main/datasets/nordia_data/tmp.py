
import pickle
import pandas as pd

# Path to your .pkl file
pkl_file = "train_data.pkl"
csv_file = "columns.csv"

# Load the pickle file
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Check if it's a DataFrame
if isinstance(data, pd.DataFrame):
    # Get column names
    columns = data.columns.tolist()

    # Save to CSV (one column name per row)
    pd.DataFrame(columns, columns=["Column"]).to_csv(csv_file, index=False)
    print(f"Column names written to {csv_file}")
else:
    print(f"Object in {pkl_file} is not a DataFrame. Type: {type(data)}")

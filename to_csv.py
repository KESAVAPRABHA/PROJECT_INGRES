import json
import pandas as pd
import os

# --- CONFIGURATION ---
# The name of your source JSON file in the 'datasets' folder
input_json_file = 'erode_taluk_data.json'
# The name of the new CSV file that will be created
output_csv_file = 'erode_taluk_data_converted.csv'

# --- SCRIPT LOGIC ---
def flatten_json(y):
    """A recursive function to flatten a nested dictionary."""
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            # We will ignore lists for this conversion to keep the table flat
            pass
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

try:
    # Construct the full path to the input file
    input_path = os.path.join('datasets', input_json_file)
    
    print(f"Reading data from '{input_path}'...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # List to hold all the flattened rows
    flattened_data = []

    # Iterate through each Taluk in the JSON file
    for taluk_name, taluk_data in data.items():
        # Flatten the nested data for the taluk
        flat_taluk = flatten_json(taluk_data)
        # Add the Taluk name as the first piece of data for that row
        flat_taluk['Taluk'] = taluk_name
        flattened_data.append(flat_taluk)
    
    # Create a pandas DataFrame from the list of flattened dictionaries
    df = pd.DataFrame(flattened_data)

    # Reorder columns to make 'Taluk' the first column for clarity
    if 'Taluk' in df.columns:
        cols = ['Taluk'] + [col for col in df if col != 'Taluk']
        df = df[cols]

    # Construct the full path for the output file
    output_path = os.path.join('datasets', output_csv_file)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    
    print(f"✅ Successfully converted the data and saved it to '{output_path}'!")
    print(f"   - Rows created: {len(df)}")
    print(f"   - Columns created: {len(df.columns)}")

except FileNotFoundError:
    print(f"❌ ERROR: The input file was not found at '{input_path}'.")
    print("Please make sure the file is in the 'datasets' folder.")
except Exception as e:
    print(f"An error occurred: {e}")
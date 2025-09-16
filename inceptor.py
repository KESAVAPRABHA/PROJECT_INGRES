import pandas as pd

try:
    # Make sure 'header=7' is correct for your file
    df = pd.read_excel("./datasets/full_data.xlsx", engine='openpyxl', header=7) 
    print("SUCCESS: The exact column names are:")
    print("-----------------------------------")
    for col in df.columns:
        print(f'"{col}"')
except Exception as e:
    print(f"An error occurred: {e}")
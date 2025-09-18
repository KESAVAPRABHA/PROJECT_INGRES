import pandas as pd

# --- CONFIGURATION ---
# Path to your source Excel file
source_excel_file = 'F:/Agentic_INGRES/datasets/full_data.xlsx'
# Path for your new, clean CSV file
output_csv_file = 'groundwater_data_cleaned.csv'

try:
    # --- 1. Read the Excel File ---
    # We specify 'header=7' to tell pandas that the real column names start on row 8
    df = pd.read_excel(source_excel_file, engine='openpyxl', header=7)
    
    # --- 2. Clean the Data ---
    # Forward-fill the merged cells for 'STATE' and 'DISTRICT'
    columns_to_fill = ["STATE", "DISTRICT"]
    for col in columns_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Remove any rows that don't have an assessment unit, as they are likely empty
    df.dropna(subset=['ASSESSMENT UNIT'], inplace=True)
    
    # --- 3. Select Only the Important Columns ---
    # Keep only the columns you need for your reports and graphs
    columns_to_keep = [
        "STATE", "DISTRICT", "ASSESSMENT UNIT", "Rainfall (mm)",
        "Stage of Ground Water Extraction (%)", 
        "Net Annual Ground Water Availability for Future Use (ham)",
        "Quality Tagging"
    ]
    
    # Ensure all columns exist before trying to select them
    final_columns = [col for col in columns_to_keep if col in df.columns]
    df_cleaned = df[final_columns]
    
    # --- 4. Save the Clean Data to a New CSV File ---
    df_cleaned.to_csv(output_csv_file, index=False)
    
    print(f"✅ Successfully transformed the data and saved it to '{output_csv_file}'!")

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at '{source_excel_file}'.")
    print("Please make sure the Excel file is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
import json
import os

# --- CONFIGURATION ---
# Define the input and output file paths
input_json_file = os.path.join('datasets', 'data_final.jsonl')
output_jsonl_file = os.path.join('datasets', 'data_final_final.jsonl')

try:
    # Read the entire standard JSON file
    with open(input_json_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    # Write each object to a new line in the .jsonl file
    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            f_out.write(json.dumps(item) + '\n')
            
    print(f"✅ Successfully converted '{input_json_file}' to '{output_jsonl_file}'!")

except FileNotFoundError:
    print(f"❌ ERROR: The input file was not found at '{input_json_file}'.")
    print("Please make sure your JSON file is named 'data.json' and is inside the 'datasets' folder.")
except Exception as e:
    print(f"An error occurred: {e}")
import pandas as pd

# Load your main CSV
# Try reading with a different separator, like a semicolon
# The 'on_bad_lines='skip'' parameter tells pandas to ignore any rows that cause errors.
df = pd.read_csv("./datasets/Erode.csv", encoding='latin-1', on_bad_lines='skip')

# Save as a JSON Lines file
# 'orient="records"' makes each row a JSON object
# 'lines=True' puts each JSON object on a new line
df.to_json("F:\\Agentic_INGRES\\datasets\\Erode.jsonl", orient="records", lines=True)

print("Successfully converted CSV to JSONL format.")
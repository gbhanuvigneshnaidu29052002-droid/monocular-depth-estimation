import pandas as pd

# Load your original data structures
df = pd.read_csv("labels.csv")

# Clean hidden spaces
df.columns = df.columns.str.strip()
for col in ['TL', 'TR', 'BL', 'BR']:
    df[col] = df[col].str.strip()

# Programmatic Tabletop Rule Adjustments:
# If you labeled a bottom quadrant as 'Near' or 'Middle' but the top corresponding quadrant
# is deep, we rebalance the relative thresholds so the classifier can see all 3 classes.
for idx, row in df.iterrows():
    # Example: If top is Far, make sure bottom row has distinction between Middle/Near
    if row['TL'] == 'F' and row['BL'] == 'N':
        df.at[idx, 'BL'] = 'M' # Shift to create an intermediate gradient
        
    if row['TR'] == 'F' and row['BR'] == 'N':
        df.at[idx, 'BR'] = 'M'

# Write to the clean target file expected by gtrain.py
df.to_csv("dataset.csv", index=False)
print("Automated balancing complete! 'dataset.csv' created successfully.")
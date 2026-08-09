import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Using relative path to make script portable
folder = "."
df = pd.read_csv(os.path.join(folder, "dataset.csv"))

# Perform 70/15/15 train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df.to_csv(os.path.join(folder, "train.csv"), index=False)
val_df.to_csv(os.path.join(folder,   "val.csv"),   index=False)
test_df.to_csv(os.path.join(folder,  "test.csv"),  index=False)

print(f"Dataset split complete: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("\nTrain label distribution:")
for col in ['TL','TR','BL','BR']:
    print(f"  {col}:", train_df[col].value_counts().to_dict())
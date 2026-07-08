import pandas as pd

# Load your processed dataset
df = pd.read_csv("data/processed/goes_labeled.csv")

# Column names in this dataset:
# time
# xrsa_flux
# xrsb_flux
# flare_class

# Save one example for each class
df[df["flare_class"] == "A"].head(500).to_csv("tests/quiet_test.csv", index=False)
df[df["flare_class"] == "C"].head(500).to_csv("tests/c_class_test.csv", index=False)
df[df["flare_class"] == "M"].head(500).to_csv("tests/m_class_test.csv", index=False)
df[df["flare_class"] == "X"].head(500).to_csv("tests/x_class_test.csv", index=False)

print("Test files created successfully.")
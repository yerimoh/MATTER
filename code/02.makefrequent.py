import pandas as pd
from collections import Counter

# Define file paths
mf_csv_path = "/mf.csv"
matname_csv_path = "/matname.csv"
o_tags_path = "/o_tags.txt"

# Define output file paths
output_mf_path = "matterm_word_frequent.csv"
output_invalid_path = "invalid_probability.csv"
output_o_tags_path = "o_word_frequent.csv"

# Load CSV files
mf_df = pd.read_csv(mf_csv_path, sep="\t", names=["Word", "Probability"], dtype={"Probability": str})
matname_df = pd.read_csv(matname_csv_path, sep="\t", names=["Word", "Probability"], dtype={"Probability": str})

# Convert all words to lowercase
mf_df["Word"] = mf_df["Word"].str.lower()
matname_df["Word"] = matname_df["Word"].str.lower()

# Check for Probability values that cannot be converted to numeric
mf_df["is_valid"] = pd.to_numeric(mf_df["Probability"], errors="coerce").notnull()
matname_df["is_valid"] = pd.to_numeric(matname_df["Probability"], errors="coerce").notnull()

# Save invalid data separately
invalid_mf = mf_df[~mf_df["is_valid"]]
invalid_matname = matname_df[~matname_df["is_valid"]]
invalid_combined = pd.concat([invalid_mf, invalid_matname])
invalid_combined.to_csv(output_invalid_path, sep="\t", index=False)
print(f"Invalid data has been saved: {output_invalid_path}")

# Process only valid data
mf_df = mf_df[mf_df["is_valid"]]
matname_df = matname_df[matname_df["is_valid"]]

# Convert Probability column to numeric
mf_df["Probability"] = pd.to_numeric(mf_df["Probability"])
matname_df["Probability"] = pd.to_numeric(matname_df["Probability"])

# Merge data
combined_df = pd.concat([mf_df, matname_df])

# Group by Word to calculate frequency and average probability
result_mf_df = combined_df.groupby("Word").agg(
    frequent=("Word", "size"),
    Probability=("Probability", "mean")
).reset_index()

# Sort by frequency in descending order
result_mf_df = result_mf_df.sort_values(by="frequent", ascending=False)

# Save the results to a CSV file
result_mf_df.to_csv(output_mf_path, sep="\t", index=False)
print(f"mf results have been saved: {output_mf_path}")

# Process o_tags.txt file
with open(o_tags_path, "r") as file:
    text = file.read()

# Split text into words and convert to lowercase
words = text.strip().lower().split()

# Count word frequencies
word_counts = Counter(words)

# Convert results to a DataFrame
result_o_tags_df = pd.DataFrame(word_counts.items(), columns=["Word", "frequent"])

# Sort by frequency in descending order
result_o_tags_df = result_o_tags_df.sort_values(by="frequent", ascending=False)

# Save the results to a CSV file
result_o_tags_df.to_csv(output_o_tags_path, sep="\t", index=False)
print(f"o_tags results have been saved: {output_o_tags_path}")

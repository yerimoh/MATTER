import os
import argparse
import pandas as pd
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

# Adjust the λ value via CLI arguments
parser = argparse.ArgumentParser(description="Material Tokenizer Training with Frequency Adjustment")
parser.add_argument("--lambda_value", type=float, default=1.0, help="Lambda value for frequency adjustment")
args = parser.parse_args()
lambda_value = args.lambda_value

# Define file paths
matterm_path = "matterm_word_frequent.csv"
o_word_path = "o_word_frequent.csv"
output_csv_path = f"adjusted_frequency_{lambda_value}.csv"
vocab_input_file = "tokenizer_input.txt"

# Set directory based on λ value
vocab_output_dir = os.path.join("vocab_output", f"vocab_{lambda_value}")
os.makedirs(vocab_output_dir, exist_ok=True)

# Load data
print("Loading data...")
matterm_df = pd.read_csv(matterm_path, sep="\t", names=["Word", "Frequent", "Probability"], header=0)
o_word_df = pd.read_csv(o_word_path, sep="\t", names=["Word", "Frequent"], header=0)

# Check for invalid data
def print_invalid_rows(df, file_name, columns):
    invalid_rows = df[[col for col in columns if col in df]].apply(lambda x: pd.to_numeric(x, errors="coerce").isna()).any(axis=1)
    if invalid_rows.any():
        print(f"Invalid rows in {file_name}:")
        print(df[invalid_rows])
    else:
        print(f"No invalid rows in {file_name}.")

print_invalid_rows(matterm_df, "matterm_word_frequent.csv", ["Frequent", "Probability"])
print_invalid_rows(o_word_df, "o_word_frequent.csv", ["Frequent"])

# Handle NaN values
matterm_df["Probability"] = pd.to_numeric(matterm_df["Probability"], errors="coerce")
matterm_df["Frequent"] = pd.to_numeric(matterm_df["Frequent"], errors="coerce")
o_word_df["Frequent"] = pd.to_numeric(o_word_df["Frequent"], errors="coerce")
matterm_df.dropna(subset=["Probability", "Frequent"], inplace=True)
o_word_df.dropna(subset=["Frequent"], inplace=True)

# Compute general word frequency distribution
general_word_frequencies = o_word_df["Frequent"].tolist()
alpha = sum([f for f in general_word_frequencies if f < 5]) / len([f for f in general_word_frequencies if f < 5])

# Function to adjust material term frequency
def adjust_frequency(freq, prob, alpha, lambda_):
    # Round P'(w|material) to two decimal places
    prob = round(prob, 2)
    
    # Handle cases where P'(w|material) is close to 1
    if prob >= 1:
        prob = 0.99  # Set a safe maximum value
    elif prob <= 0:
        prob = 0.01  # Set a safe minimum value
    
    # Compute adjusted frequency
    emphasized_factor = lambda_ * (prob / (1 - prob))
    return freq + emphasized_factor

# Store results
results = []

# Process material terms
print("Processing material terms...")
for _, row in tqdm(matterm_df.iterrows(), total=len(matterm_df), desc="Processing material terms"):
    word, freq, prob = row["Word"], row["Frequent"], row["Probability"]
    adjusted_freq = adjust_frequency(freq, prob, alpha, lambda_value)
    results.append([str(word), "material term", freq, adjusted_freq])  # Convert word to string

# Process general words
print("Processing general words...")
for _, row in tqdm(o_word_df.iterrows(), total=len(o_word_df), desc="Processing general words"):
    word, freq = row["Word"], row["Frequent"]
    results.append([str(word), "general word", freq, freq])  # Convert word to string

# Save adjusted frequencies to CSV
print("Saving adjusted frequencies to CSV...")
output_df = pd.DataFrame(results, columns=["Word", "Tag", "Original Frequency", "Adjusted Frequency"])
output_df.to_csv(output_csv_path, index=False)

# Duplicate words based on adjusted frequency
print("Duplicating words based on adjusted frequencies...")
with open(vocab_input_file, "w") as f:
    for _, row in tqdm(output_df.iterrows(), total=len(output_df), desc="Writing duplicated words"):
        word, adjusted_freq = row["Word"], row["Adjusted Frequency"]
        for _ in range(int(round(adjusted_freq))):  # Duplicate based on adjusted frequency
            f.write(str(word) + "\n")  # Convert word to string

# Train WordPiece tokenizer
print("Training tokenizer...")
tokenizer = BertWordPieceTokenizer(strip_accents=True, lowercase=True)
tokenizer.train(
    files=[vocab_input_file],
    vocab_size=31090,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

# Save tokenizer
tokenizer.save_model(vocab_output_dir)

print(f"Vocab saved to {vocab_output_dir}/vocab.txt")

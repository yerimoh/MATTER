import os
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForTokenClassification
from tqdm import tqdm

# path
model_path = 'MatTermDetector Path'
tokenizer_path = 'FIXPATH/matbert-base-cased'
input_file = 'TARGT_CORPUS.txt'
output_directory = './'
batch_size = 256 * 16 # batch

# Load
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, do_lower_case=False)
model = BertForTokenClassification.from_pretrained(model_path).half()  # FP16 

# GPU 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[2])  # GPU 
model.to(device)

# Label Decoding Map
label_map = {0: "O", 1: "B-matname", 2: "I-matname", 3: "B-mf", 4: "I-mf"}


def process_batch(batch_texts, tokenizer, model, device):
    """
    Tokenize sentences in batches and enter them into the model to calculate the probability.
    """
    tokenized = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=2)  # (batch_size, seq_len, num_labels)

    return tokenized, probabilities

def determine_final_label(batch_texts, tokenized, probabilities, label_map):
    """
    Determined the final label for words in each sentence.
    """
    results = []
    for i, text in enumerate(batch_texts):
        words = text.split() 
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][i].tolist())
        probs = probabilities[i]  # (seq_len, num_labels)

        current_word = ""
        current_label = "O"
        current_prob = []
        word_index = 0  # 공백 기준 단어 인덱스

        for token, prob in zip(tokens, probs):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if token.startswith("##"):
                current_word += token[2:]
                current_prob.append(prob.max().item())
            else:
                if current_word:
                    avg_prob = sum(current_prob) / len(current_prob)
                    results.append((current_word, current_label, avg_prob))
                    current_word = "" 
                    current_prob = []

                if word_index < len(words) and token == words[word_index]:
                    current_word = words[word_index]
                    word_index += 1
                else:
                    current_word = token

                current_label = label_map[prob.argmax().item()]
                current_prob = [prob.max().item()]

        if current_word:
            avg_prob = sum(current_prob) / len(current_prob)
            results.append((current_word, current_label, avg_prob))

    return results




def save_results(results, mf_writer, matname_writer, o_writer):
    """
    Save the final result.
    """
    for word, label, prob in results:
        if label == "O":
            o_writer.write(f"{word}\n")  
        elif label in ["B-mf", "I-mf"]:
            mf_writer.writerow([word, prob])
        elif label in ["B-matname", "I-matname"]:
            matname_writer.writerow([word, prob])






with open(os.path.join(output_directory, "mf.csv"), "w", newline="") as mf_file, \
     open(os.path.join(output_directory, "matname.csv"), "w", newline="") as matname_file, \
     open(os.path.join(output_directory, "o_tags.txt"), "w") as o_file:

    mf_writer = csv.writer(mf_file, delimiter='\t')
    matname_writer = csv.writer(matname_file, delimiter='\t')

    mf_writer.writerow(["Word", "Probability"])
    matname_writer.writerow(["Word", "Probability"])

    with open(input_file, 'r') as file:
        lines = file.readlines()
        total_lines = len(lines)

        with tqdm(total=total_lines, desc="Extracting P(w|material) using ME", unit="lines") as progress_bar:
            for i in range(0, total_lines, batch_size):
                batch_texts = [line.strip() for line in lines[i:i + batch_size]]
                tokenized, probabilities = process_batch(batch_texts, tokenizer, model, device)
                results = determine_final_label(batch_texts, tokenized, probabilities, label_map)
                save_results(results, mf_writer, matname_writer, o_file)
                progress_bar.update(len(batch_texts))

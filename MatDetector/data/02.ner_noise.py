import json
import random
import re
from tqdm import tqdm
from transformers import AutoTokenizer

# Load file name and tokenizer
file = 'nose_data'
tokenizer = AutoTokenizer.from_pretrained("/mnt/user25/Material_tokenizer/02.makeNER_ver2/matbert-base-cased")

# Periodic table element symbols
ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
    'Lv', 'Ts', 'Og'
]

# Noise injection functions
def random_char_insert(token):
    """
    Insert a random character into the token
    """
    pos = random.randint(0, len(token) - 1)
    random_char = chr(random.randint(97, 122))  # a-z
    return token[:pos] + random_char + token[pos:]

def add_brackets_no_space(token):
    """
    Add brackets to digits (no spaces)
    """
    return ''.join(f"({c})" if c.isdigit() else c for c in token)

def add_brackets_with_space(token):
    """
    Add brackets to digits (with spaces)
    """
    return ''.join(f" ( {c} ) " if c.isdigit() else c for c in token)

def insert_punctuation(token):
    """
    Insert a punctuation mark
    """
    pos = random.randint(0, len(token) - 1)
    return token[:pos] + '.' + token[pos:]

def case_error(token):
    """
    Introduce a casing error
    """
    pos = random.randint(0, len(token) - 1)
    if token[pos].isalpha():
        return token[:pos] + token[pos].swapcase() + token[pos + 1:]
    return token

def number_position_error(token):
    """
    Introduce a digit position error
    """
    pos = random.randint(0, len(token) - 1)
    if token[pos].isdigit():
        return token[:pos] + token[pos] + token[pos:]
    return token

def detailed_formula(token):
    """
    Detailed form of characters and digits
    """
    return ''.join(f"{c}({c})" if c.isdigit() else f"{c}" for c in token)

def special_char_insert(token):
    """
    Insert a special character
    """
    pos = random.randint(0, len(token))
    special_char = random.choice(['*', '&', '%', '$', '#'])
    return token[:pos] + special_char + token[pos:]

# List of noise generator functions
noise_functions = [
    random_char_insert,
    add_brackets_no_space,
    add_brackets_with_space,
    insert_punctuation,
    case_error,
    number_position_error,
    #detailed_formula,
    special_char_insert
]

def apply_mf_rules(token):
    """
    Apply noise rules for MF entities
    """
    noise_func = random.choice(noise_functions)
    return noise_func(token)

def apply_custom_rules_1(token):
    """
    Apply noise rules (case 1)
    """
    token = re.sub(r'-(?! )', ' - ', token)
    token = re.sub(r'(?<! )-', ' - ', token)
    token = re.sub(r'\(', ' ( ', token)
    token = re.sub(r'\)', ' ) ', token)
    token = re.sub(r'\[', ' [ ', token)
    token = re.sub(r'\]', ' ] ', token)
    return token

def apply_custom_rules_2(token):
    """
    Apply noise rules (case 2)
    """
    return re.sub(r'\d', 'x', token)

def apply_custom_rules_3(token):
    """
    Apply noise rules (case 3)
    """
    token = re.sub(r'-(?! )', ' - ', token)
    token = re.sub(r'(?<! )-', ' - ', token)
    token = re.sub(r'\(', ' ( ', token)
    token = re.sub(r'\)', ' ) ', token)
    token = re.sub(r'\[', ' [ ', token)
    token = re.sub(r'\]', ' ] ', token)
    return re.sub(r'\d', 'x', token)

def apply_all_mf_rules(token):
    """
    Apply all noise rules to MF entity
    """
    noisy_tokens = []
    for noise_func in noise_functions:
        noisy_tokens.append(noise_func(token))
    return noisy_tokens

def process_data_with_custom_rules(data, apply_custom_rules, output_file):
    # Add noise and re-tokenize for each token
    noisy_data = []
    for item in tqdm(data, desc="Processing items"):
        for noise_idx in range(len(noise_functions)):  # Iterate through each noise function
            new_item = {
                "date": item["date"],
                "cmpdname": item["cmpdname"],
                "doi": item["doi"],
                "text": [],
                "labels": []
            }
            new_text = []
            new_labels = []
            temp_token = ""
            temp_labels = []

            for token, label in zip(item["text"], item["labels"]):
                if label.startswith("B-mf") or label.startswith("B-cmpdsynonym") or label.startswith("B-cmpdname") or label.startswith("B-iupacname"):
                    if temp_token:
                        # Tokenize and label previously collected entity
                        tokenized = tokenizer.tokenize(temp_token)
                        new_text.extend(tokenized)
                        new_labels.extend([temp_labels[0]] + [temp_labels[0].replace("B-", "I-")] * (len(tokenized) - 1))
                        temp_token = ""
                        temp_labels = []
                    temp_token = token
                    temp_labels = [label]
                elif label.startswith("I-mf") or label.startswith("I-cmpdsynonym") or label.startswith("I-cmpdname") or label.startswith("I-iupacname"):
                    temp_token += token.replace("##", "")
                    temp_labels.append(label)
                else:
                    if temp_token:
                        # Apply appropriate noise rule depending on entity type
                        entity_type = temp_labels[0][2:]
                        if entity_type in ["mf", "cmpdsynonym", "cmpdname", "iupacname"]:
                            if entity_type == "mf":
                                noisy_token = apply_all_mf_rules(temp_token)[noise_idx]
                            else:
                                noisy_token = apply_custom_rules(temp_token)
                            # Tokenize and label noisy entity
                            tokenized = tokenizer.tokenize(noisy_token)
                            new_text.extend(tokenized)
                            new_labels.extend([temp_labels[0]] + [temp_labels[0].replace("B-", "I-")] * (len(tokenized) - 1))
                        temp_token = ""
                        temp_labels = []
                    new_text.append(token)
                    new_labels.append(label)

            if temp_token:
                # Process remaining entity
                entity_type = temp_labels[0][2:]
                if entity_type in ["mf", "cmpdsynonym", "cmpdname", "iupacname"]:
                    if entity_type == "mf":
                        noisy_token = apply_all_mf_rules(temp_token)[noise_idx]
                    else:
                        noisy_token = apply_custom_rules(temp_token)
                    tokenized = tokenizer.tokenize(noisy_token)
                    new_text.extend(tokenized)
                    new_labels.extend([temp_labels[0]] + [temp_labels[0].replace("B-", "I-")] * (len(tokenized) - 1))

            new_item["text"] = new_text
            new_item["labels"] = new_labels

            # Preserve cmpdsynonym, cmpdname, iupacname fields
            for key in ["cmpdsynonym", "cmpdname", "iupacname"]:
                if key in item:
                    new_item[key] = item[key]

            noisy_data.append(new_item)

    # Save results to output JSON file
    with open(output_file, 'w') as f:
        json.dump(noisy_data, f, indent=4)

    print(f"Noisy data has been saved to {output_file}")


# Load input JSON file
input_file = f'/mnt2/user25/02.makeNER_ver2/03.sum_non_noise/non_noise_full_clean.json'
with open(input_file, 'r') as f:
    data = json.load(f)

# Process data for each case
process_data_with_custom_rules(data, apply_custom_rules_1, f'/mnt2/user25/02.makeNER_ver2/04.noise_ver1/{file}_case1.json')
process_data_with_custom_rules(data, apply_custom_rules_2, f'/mnt2/user25/02.makeNER_ver2/04.noise_ver1/{file}_case2.json')
process_data_with_custom_rules(data, apply_custom_rules_3, f'/mnt2/user25/02.makeNER_ver2/04.noise_ver1/{file}_case3.json')

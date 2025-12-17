# MATTER

### ✨ [ACL 2025] Incorporating Domain Knowledge into Materials Tokenization
### 🎇 SAC Highlights
<div align="center">
  <img src="src/main.jpg" alt="Main Figure" width="600"/>
</div>


----

# MatDetector

If you only want to extract material concepts (material term or material formula) using **MatDetector**, please follow the steps below.

**1.** Download the [MatDetector](https://huggingface.co/yerim0210/MatDetector/blob/main/README.md) checkpoint   
**2.** Use the following code to detect material concepts:


```python
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForTokenClassification
from tqdm import tqdm

model_path = MatDetector_ckp
# you can download matbert at https://github.com/lbnlp/MatBERT
tokenizer_path = '/matbert-base-cased'
input_file = 'TARGET.txt'
output_directory = './'


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, do_lower_case=False)
model = BertForTokenClassification.from_pretrained(model_path).half()  

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() 

label_map = {0: "O", 1: "B-matname", 2: "I-matname", 3: "B-mf", 4: "I-mf"}

def process_single_word(word, tokenizer, model, device):
    tokenized = tokenizer(word, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=2)  # (batch_size=1, seq_len, num_labels)

    return tokenized, probabilities


def determine_label(tokenized, probabilities, label_map):
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0].tolist())
    probs = probabilities[0]  # (seq_len, num_labels)

    token_labels = []
    for token, prob in zip(tokens, probs):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_token = token[2:] if token.startswith("##") else token
        max_label = prob.argmax().item()
        label_name = label_map[max_label]

        token_labels.append(label_name)

    label_counts = {}
    for label in token_labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    final_label = max(label_counts, key=label_counts.get) if label_counts else "O"
    
    return final_label


with open(os.path.join(output_directory, "mf.txt"), "w") as mf_file, \
     open(os.path.join(output_directory, "matname.txt"), "w") as matname_file, \
     open(os.path.join(output_directory, "o_tags.txt"), "w") as o_file:

    with open(input_file, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]  
        total_lines = len(lines)

        with tqdm(total=total_lines, desc="Processing words", unit="words") as progress_bar:
            for original_word in lines:
                tokenized, probabilities = process_single_word(original_word, tokenizer, model, device)
                final_label = determine_label(tokenized, probabilities, label_map)

                if final_label == "O":
                    o_file.write(f"{original_word}\n")
                elif final_label in ["B-mf", "I-mf"]:
                    mf_file.write(f"{original_word}\n")
                elif final_label in ["B-matname", "I-matname"]:
                    matname_file.write(f"{original_word}\n")

                progress_bar.update(1)

print("Processing completed. Files saved as mf.txt, matname.txt, and o_tags.txt.")

```


----

# MATTER Tokenization





## STEP1 : Extract Material Concepts with MatDetector
The training data for **MatDetector**, which extracts material concepts and their probabilities, is only shared as DOIs due to copyright issues. 
However, with a free API key from Semantic Scholar, you can crawl the papers using those DOIs to get the training data.

We’ve included code for crawling, NER, and adding noise to make the data more robust. Using this, you can build an NER dataset and train MatDetector with the provided training script.
All data-related resources are stored in the following folder:

```python
## doi
MatDetector/data/doi.zip

## data create
MatDetector/data/

## training MatDetector
MatDetector
```

To make things easier, we also released a pretrained MatDetector and a demo. If you're using the MATTER framework, you can use this model and the released tokenization code to build a tokenizer for your own corpus.

```python
MatDetector/ckp
```




## STEP2: create MATTER tokenization 



### Installing requirements
```python
cd code
bash install_requirements.sh
```


For MATTER Tokenization training, use this code.     
you can choose lambda in there.
```python
run_MATTER.sh
```



## STEP3: Training with MATTER tokenization

```python
train.sh
```




## Evaluation

### Generation

```python
cd eval/generation/
bash run.sh
```

### Clssification


```python
cd eval/classification/
```

#### NER
```python
cd ner
# If the model uses a BPE-based tokenizer, run.sh with ner_BPE.py
# otherwise, run.sh with ner.py
run.sh
```

#### RC
```python
cd relation_classification
# If the model uses a BPE-based tokenizer, run.sh with relation_classification_BPE.py
# otherwise, run.sh with relation_classification.py
run.sh
```

#### CLS

```python
cd cls
run.sh
```

----

# Citation


```
Yerim oh et al., Incorporating Domain Knowledge into Materials Tokenization, ACL 2025
```


-----



If you have any questions, feel free to reach out to me at **yerim0210@korea.ac.kr**. I’ll get back to you as quickly as possible.


import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments, default_data_collator
from transformers.integrations import TensorBoardCallback
from datasets import load_metric
import numpy as np
from transformers import AutoTokenizer

model_path = '/matbert-base-cased'
data_path = '/data.json'

batch = 32


class NERDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=128):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.max_len = max_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']

        # Ensure tokenized text and labels are the same length
        assert len(text) == len(labels), "Mismatch between text and labels length"

        # Change the label to "O" for stop words tagged as "B-matname" or "B-mf"
        labels = [
            "O" 
            for word, label in zip(text, labels)
        ]

        # Convert subword tokens to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        
        # Truncate and pad input IDs and labels to max length
        input_ids = input_ids[:self.max_len]
        attention_mask = [1] * len(input_ids)
        labels = labels[:self.max_len]

        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        labels = labels + ([-100] * padding_length)

        encodings = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor([self.label2id(label) for label in labels])
        }
        '''
        # Print the first data sample if it contains at least one non-"O" label
        if idx == 30 and any(label != "O" for label in item['labels']):
            tokens = text + ["[PAD]"] * padding_length
            print("Token: Label -> Encoded Label")
            for token, label, encoded_label in zip(tokens, labels, encodings['labels']):
                print(f"{token}: {label} -> {encoded_label}")
        '''


        return encodings

    def label2id(self, label):
        label_map = {
            "O": 0, 
            "B-matname": 1, 
            "I-matname": 2, 
            "B-mf": 3, 
            "I-mf": 4, 
        }
        if label in ["B-cmpdname", "B-cmpdsynonym", "B-iupacname"]:
            return label_map["B-matname"]
        elif label in ["I-cmpdname", "I-cmpdsynonym", "I-iupacname"]:
            return label_map["I-matname"]
        elif label in ["I-"]:
            return label_map["I-mf"]
        else:
            return label_map.get(label, -100)


tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
dataset = NERDataset(tokenizer, data_path, max_len=128)
train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=default_data_collator)
model = BertForTokenClassification.from_pretrained(model_path, num_labels=5)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_id for label_id in label if label_id != -100] for label in labels]
    true_predictions = [
        [pred_id for (pred_id, label_id) in zip(prediction, label) if label_id != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir='/ckpt',
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch,
    num_train_epochs=60,
    weight_decay=0.01,
    logging_dir='/logs',
    logging_steps=10,
    save_steps=100000,
    dataloader_num_workers=4,
    report_to="tensorboard",
    fp16=True,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
    callbacks=[TensorBoardCallback()],
)

trainer.train()

trainer.save_model('./matdetector')

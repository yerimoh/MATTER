import os
from pathlib import Path
import pickle
import sys
sys.path.append('..')
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from datasets import load_dataset
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from normalize_text import normalize
from sklearn.metrics import precision_recall_fscore_support

import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


parser = ArgumentParser()
parser.add_argument('--model_name', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--preds_save_dir', default=None, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--seeds', nargs='+', default=None, type=int)
parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
args = parser.parse_args()

if args.model_name == 'scibert':
    model_name = 'allenai/scibert_scivocab_uncased'
    to_normalize = False
elif args.model_name == 'matscibert':
    model_name = 'm3rg-iitd/matscibert'
    to_normalize = True
elif args.model_name == 'bert':
    model_name = 'bert-base-uncased'
    to_normalize = False
elif args.model_name == 'MatiSciBERT_20k_orgin':
    model_name = '/mnt2/user25/orgin/checkpoint-200000'
    to_normalize = True
elif args.model_name == 'lamda2_1.0':
    model_name = '/home/user25/WorkSpace/MatTokenization/05.pretraining/MatSciBERT/pretraining/last/lamda2/1.0/checkpoint-100000'
    to_normalize = True
elif args.model_name == 'sage':
    model_name = '/home/user25/WorkSpace/MatTokenization/05.pretraining/MatSciBERT/ner/model_innal/sage/checkpoint-100000'
    to_normalize = True
elif args.model_name == 'wp':
    model_name = '/home/user25/WorkSpace/MatTokenization/05.pretraining/MatSciBERT/ner/model_innal/WP/checkpoint-100000'
    to_normalize = True
elif args.model_name == 'PickyBPE':
    model_name = '/home/user25/WorkSpace/MatTokenization/05.pretraining/MatSciBERT/pretraining/last/picky_0.9/resultt/checkpoint-100000'
    to_normalize = True
else:
    raise NotImplementedError



#####################
# save result
txt_name = f'128_{args.model_name}_{args.seeds}'
#####################


model_revision = 'main'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None

if args.seeds is None:
    args.seeds = [0, 1, 2]
if args.lm_lrs is None:
    args.lm_lrs = [2e-5, 3e-5, 5e-5]

dataset_dir = 'datasets/glass_non_glass'
data_files = {split: os.path.join(dataset_dir, f'{split}.csv') for split in ['train', 'val', 'test']}
datasets = load_dataset('csv', data_files=data_files, cache_dir=cache_dir)

label_list = datasets['train'].unique('Label')
num_labels = len(label_list)

max_seq_length = 512

config_kwargs = {
    'num_labels': num_labels,
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': max_seq_length
}

from pathlib import Path

from transformers import PreTrainedTokenizerFast
from pathlib import Path

class CustomTokenizerWrapper(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_object, **kwargs):
        super().__init__(tokenizer_object=tokenizer_object, **kwargs)
        self.tokenizer_object = tokenizer_object

    def save_vocabulary(self, save_directory, filename_prefix=None):
        # 저장 경로 생성
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # vocab.json 및 merges.txt 저장 경로
        vocab_path = save_dir / (filename_prefix + "-vocab.json" if filename_prefix else "vocab.json")
        merges_path = save_dir / (filename_prefix + "-merges.txt" if filename_prefix else "merges.txt")

        # 내부 tokenizer_object의 저장 메서드 호출
        self.tokenizer_object.save(str(vocab_path), pretty=False)  # 두 번째 인수를 bool로 설정
        self.tokenizer_object.save(str(merges_path), pretty=False)

        return str(vocab_path), str(merges_path)


if args.model_name == 'PickyBPE':
    # Load the PickyBPE tokenizer
    tokenizer_path = "/home/user25/WorkSpace/MatTokenization/05.pretraining/MatSciBERT/pretraining/last/picky_0.9/result2"
    tokenizer = CustomTokenizerWrapper(
        tokenizer_object=ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_path, "vocab.json"),
            merges_filename=os.path.join(tokenizer_path, "merges.txt")
        )
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)



def preprocess_function(examples):
    if to_normalize:
        examples['Abstract'] = list(map(normalize, examples['Abstract']))
    result = tokenizer(examples['Abstract'], padding=False, max_length=max_seq_length, truncation=True)
    result['label'] = [l for l in examples['Label']]
    return result


tokenized_datasets = datasets.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['val']
test_dataset = tokenized_datasets['test']


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    prec, recall, fscore, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    return {
        'accuracy': (preds == p.label_ids).astype(np.float32).mean().item(),
        'precision': prec,
        'recall': recall,
        'fscore': fscore
    }


metric_for_best_model = 'accuracy'

best_lr = None
best_val = 0
best_val_acc_list = None
best_test_acc_list = None


for lr in args.lm_lrs:

    print(f'lr: {lr}')
    val_acc, test_acc = [], []
    
    for SEED in args.seeds:
        
        print(f'SEED: {SEED}')
        
        #torch.set_deterministic(True)
        torch.use_deterministic_algorithms(False)

        torch.backends.cudnn.benchmark = False
        set_seed(SEED)
        
        training_args = TrainingArguments(
            num_train_epochs=10,
            output_dir=output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            learning_rate=lr,
            seed=SEED
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=None,
        )

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': args.non_lm_lr},
            {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': lr}
        ]
        optimizer_kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
        )

        train_result = trainer.train()
        print(train_result)
        val_result = trainer.evaluate()
        print(val_result)
        test_result = trainer.evaluate(test_dataset)
        print(test_result)
        f = open(f"./{txt_name}.txt",'a')
        f.write(f'\ntrain_result: {train_result}')
        f.write(f'\nval_result: {val_result}')
        f.write(f'\ntest_result: {test_result}')
        f.close()
        val_acc.append(val_result[f'eval_{metric_for_best_model}'])
        test_acc.append(test_result[f'eval_{metric_for_best_model}'])

        if preds_save_dir:
            val_preds = trainer.predict(val_dataset).predictions
            test_preds = trainer.predict(test_dataset).predictions

            for split, preds in zip(['val', 'test'], [val_preds, test_preds]):
                file_path = os.path.join(preds_save_dir, f'glass_non_glass/{split}_{args.model_name}_{lr}_{SEED}.pkl')
                ensure_dir(os.path.dirname(file_path))
                pickle.dump(preds, open(file_path, 'wb'))
    
    if np.mean(val_acc) > best_val:
        best_val = np.mean(val_acc)
        best_lr = lr
        best_val_acc_list = val_acc
        best_test_acc_list = test_acc


print(args.model_name)
print('best_lr: ', best_lr)
print('best_val: ', best_val)
print('best_val_acc_list: ', best_val_acc_list)
print('best_test_acc_list: ', best_test_acc_list)


f = open(f"{txt_name}.txt",'a')
print('here')
#print(test_result)

f.write(args.model_name )
f.write(f'\nSEED: {SEED}')
f.write(f'\nbest_val: {best_val}')
f.write(f'\nbest_val_acc_list: {best_val_acc_list}')
f.write(f'\nbest_test_acc_list: {best_test_acc_list}')
f.close()
# MATTER
This repository contains code for the paper "Incorporating Domain Knowledge into Materials Tokenization"

----

## Training



### Installing requirements
```python
cd code
bash install_requirements.sh
```


### Tokenization Training
For MATTER Tokenization training, use this code.     
you can choose lambda in there.
```python
run_MATTER.sh
```

### Training model

```python
train.sh
```


----

## Eveluation

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


# copied for reference from:
# https://huggingface.co/docs/transformers/tasks/token_classification
# (will be removed later)

from datasets import load_dataset
from transformers import \
    AutoTokenizer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification, TrainingArguments, Trainer


wnut = load_dataset('wnut_17')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    print(type(examples), type(tokenized_inputs))
    
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=14)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

#model = AutoModelForTokenClassification.from_pretrained('model')

tok = tokenizer.convert_ids_to_tokens(tokenized_wnut['train']['input_ids'][-1])
lab = tokenized_wnut['train']['labels'][-1]
print(list(zip(tok, lab)))


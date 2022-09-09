import argparse
from collections import defaultdict
import csv

import pyarrow as pa

from datasets import load_dataset, Dataset, ClassLabel, Sequence
from transformers import \
    AutoTokenizer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification, TrainingArguments, Trainer


def load_dataset_from_conll(filename):
    # TODO include paragraph breaks?
    doc_ids, s_ids, w_ids, tokens = [], [], [], []
    cur_doc_id, cur_s_ids, cur_w_ids, cur_tokens = None, [], [], []
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row['articleId'] != cur_doc_id:
                if cur_tokens:
                    doc_ids.append(cur_doc_id)
                    s_ids.append(cur_s_ids)
                    w_ids.append(cur_w_ids)
                    tokens.append(cur_tokens)
                    cur_s_ids, cur_w_ids, cur_tokens = [], [], []
                cur_doc_id = row['articleId']
            cur_s_ids.append(int(row['sentenceId']))
            cur_w_ids.append(int(row['wordId']))
            cur_tokens.append(row['word'])
        if cur_tokens:
            doc_ids.append(cur_doc_id)
            s_ids.append(cur_s_ids)
            w_ids.append(cur_w_ids)
            tokens.append(cur_tokens)
    return Dataset(pa.table([doc_ids, s_ids, w_ids, tokens],
                            names=['doc_id', 's_ids', 'w_ids', 'tokens']))

def load_annotations(filename):
    doc_id, anns = None, []
    results = defaultdict(list)
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            results[row['articleId']].append(row)
    return results

def add_quote_annotations(docs, anns):
    cl = Sequence(ClassLabel(
             num_classes=5,
             names=['O', 'B-DIRECT', 'I-DIRECT', 'B-INDIRECT', 'I-INDIRECT']))
    labels_per_doc = []
    for d in docs:
        labels = ['O' for t in d['tokens']]
        if d['doc_id'] in anns:
            tok_inv_dict = { (d['s_ids'][i], d['w_ids'][i]) : i \
                             for i in range(len(d['s_ids'])) }
            for a in anns[d['doc_id']]:
                i = tok_inv_dict[(int(a['startSentenceId']), int(a['startWordId']))]
                lbl_direct = 'DIRECT' if a['direct'] == 'true' else 'INDIRECT'
                labels[i] = 'B-{}'.format(lbl_direct)
                i += 1
                while d['s_ids'][i] < int(a['endSentenceId']) \
                      or d['w_ids'][i] < int(a['endWordId']):
                    labels[i] = 'I-{}'.format(lbl_direct)
                    i += 1
        labels_per_doc.append(labels)
    docs = docs.add_column('quote-tags', labels_per_doc)\
               .cast_column('quote-tags', cl)
    return docs

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['quote-tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
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
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def parse_arguments():
    parser = argparse.ArgumentParser(description='Rule-based quote detection.')
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-a', '--annotations-file', metavar='FILE')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    docs = load_dataset_from_conll(args.input_file)
    anns = load_annotations(args.annotations_file)
    docs = add_quote_annotations(docs, anns)
    
    tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
    model = AutoModelForTokenClassification.from_pretrained(
                'TurkuNLP/bert-base-finnish-cased-v1',
                num_labels=5)
    
    tokenized_docs = docs.map(tokenize_and_align_labels, batched=True)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_docs,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    
    # FIXME training very slow and takes huge amounts of memory -> too long sequences?
    

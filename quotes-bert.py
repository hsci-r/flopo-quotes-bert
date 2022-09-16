import argparse
from collections import defaultdict
import csv
import re
import sys

import numpy as np
import pyarrow as pa
import tqdm

from datasets import load_dataset, Dataset, ClassLabel, Sequence
import torch
from transformers import \
    AutoTokenizer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification, TrainingArguments, Trainer


LABELS = ['O', 'B-DIRECT', 'I-DIRECT', 'B-INDIRECT', 'I-INDIRECT']


def load_dataset_from_conll(filename):
    pat = re.compile(r'SpacesAfter=.*\\n.*')
    doc_ids, par_ids, s_ids, w_ids, tokens = [], [], [], [], []
    cur_doc_id, cur_par_id, cur_s_ids, cur_w_ids, cur_tokens = None, None, [], [], []
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row['articleId'] != cur_doc_id or row['paragraphId'] != cur_par_id:
                if cur_tokens:
                    doc_ids.append(cur_doc_id)
                    par_ids.append(int(cur_par_id))
                    s_ids.append(cur_s_ids)
                    w_ids.append(cur_w_ids)
                    tokens.append(cur_tokens)
                    cur_s_ids, cur_w_ids, cur_tokens = [], [], []
                cur_doc_id = row['articleId']
                cur_par_id = row['paragraphId']
            cur_s_ids.append(int(row['sentenceId']))
            cur_w_ids.append(int(row['wordId']))
            cur_tokens.append(row['word'])
            if pat.search(row['misc']) and tokens:
                doc_ids.append(cur_doc_id)
                par_ids.append(int(cur_par_id))
                s_ids.append(cur_s_ids)
                w_ids.append(cur_w_ids)
                tokens.append(cur_tokens)
                cur_s_ids, cur_w_ids, cur_tokens = [], [], []
        if cur_tokens:
            doc_ids.append(cur_doc_id)
            par_ids.append(int(cur_par_id))
            s_ids.append(cur_s_ids)
            w_ids.append(cur_w_ids)
            tokens.append(cur_tokens)
    return Dataset(pa.table([doc_ids, par_ids, s_ids, w_ids, tokens],
                            names=['doc_id', 'par_id', 's_ids', 'w_ids', 'tokens']))

def load_annotations(filename):
    doc_id, anns = None, []
    results = defaultdict(list)
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            results[row['articleId']].append(row)
    return results

def add_quote_annotations(docs, anns):
    cl = Sequence(ClassLabel(num_classes=len(LABELS), names=LABELS))
    labels_per_doc = []
    for d in docs:
        labels = ['O' for t in d['tokens']]
        if d['doc_id'] in anns:
            tok_inv_dict = { (d['s_ids'][i], d['w_ids'][i]) : i \
                             for i in range(len(d['s_ids'])) }
            for a in anns[d['doc_id']]:
                start_idx = (int(a['startSentenceId']), int(a['startWordId']))
                i = None
                if start_idx in tok_inv_dict:
                    i = tok_inv_dict[start_idx]
                elif (d['s_ids'][0] > int(a['startSentenceId']) \
                      or (d['s_ids'][0] == int(a['startSentenceId']) \
                          and d['w_ids'][0] > int(a['startWordId']))) \
                      and (d['s_ids'][0] < int(a['endSentenceId']) \
                           or (d['s_ids'][0] == int(a['endSentenceId']) \
                               and d['w_ids'][0] < int(a['endWordId']))):
                    i = 0
                else:
                    continue
                lbl_direct = 'DIRECT' if a['direct'] == 'true' else 'INDIRECT'
                labels[i] = 'B-{}'.format(lbl_direct)
                i += 1
                while i < len(d['tokens']) \
                      and (d['s_ids'][i] < int(a['endSentenceId']) \
                           or d['w_ids'][i] < int(a['endWordId'])):
                    labels[i] = 'I-{}'.format(lbl_direct)
                    i += 1
        labels_per_doc.append(labels)
    docs = docs.add_column('quote-tags', labels_per_doc)\
               .cast_column('quote-tags', cl)
    return docs

def tokenize_and_align_labels(examples, max_length):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=max_length)
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
    parser = argparse.ArgumentParser(description='Quote detection using BERT.')
    parser.add_argument('cmd', choices=['train', 'run'])
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-m', '--model-dir', metavar='DIR')
    parser.add_argument('-a', '--annotations-file', metavar='FILE')
    parser.add_argument('-n', '--max-length', type=int, default=20,
                        help='maximum sequence length')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.cmd == 'train':

        docs = load_dataset_from_conll(args.input_file)
        anns = load_annotations(args.annotations_file)
        docs = add_quote_annotations(docs, anns)
    
        tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
        model = AutoModelForTokenClassification.from_pretrained(
                    'TurkuNLP/bert-base-finnish-cased-v1',
                    num_labels=5)
        
        tokenized_docs = docs.map(
            lambda x: tokenize_and_align_labels(x, args.max_length),
            batched=True)
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding='max_length')
        training_args = TrainingArguments(
            output_dir=args.model_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=args.epochs,
            weight_decay=0.01)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_docs,
            eval_dataset=tokenized_docs.select(np.random.choice(len(tokenized_docs), int(len(tokenized_docs)*0.1))),
            tokenizer=tokenizer,
            data_collator=data_collator)
        trainer.train()
        
        model.save_pretrained(args.model_dir)

    elif args.cmd == 'run':
        
        docs = load_dataset_from_conll(args.input_file)
        tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=['articleId', 'startSentenceId', 'startWordId',
                        'endSentenceId', 'endWordId', 'direct'])
        writer.writeheader()
        for d in tqdm.tqdm(docs):
            toks = tokenizer(d['tokens'], return_tensors='pt', padding=True, is_split_into_words=True)
            pred = model(**toks)
            results = torch.argmax(pred['logits'], 2).flatten()
            j = 0
            while j < results.size()[0]:
                if LABELS[results[j]].startswith('B-'):
                    word_ids = toks.word_ids()
                    k = j
                    while k < results.size()[0] and word_ids[k] is None:
                        k += 1
                    start_idx = word_ids[k]
                    i_lab = LABELS[results[j]+1]
                    direct = (LABELS[results[j]] == 'B-DIRECT')
                    while j < results.size()[0]-1 and LABELS[results[j+1]] == i_lab:
                        j += 1
                    k = j
                    while k > 0 and word_ids[k] is None:
                        k -= 1
                    end_idx = word_ids[k]
                    # TODO error / warning if there is sth wrong with indices
                    if start_idx is not None and end_idx is not None and end_idx >= start_idx:
                        writer.writerow({
                            'articleId': d['doc_id'], 
                            'startSentenceId': d['s_ids'][start_idx],
                            'startWordId': d['w_ids'][start_idx],
                            'endSentenceId': d['s_ids'][end_idx], 
                            'endWordId': d['w_ids'][end_idx], 
                            'direct': str(direct).lower()
                        })
                j += 1


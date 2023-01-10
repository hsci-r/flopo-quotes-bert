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
    AutoTokenizer, BertPreTrainedModel, BertModel, \
    DataCollatorForTokenClassification, TrainingArguments, Trainer, \
    AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

# TODO
# - load the corpus from CoNLL (paragraph-wise)
# - add the entity annotations (paragraph-wise without a problem)
# - add the quote annotations
#   - process the paragraphs by document
#   - translate the pairs (sentenceId, wordId) to (paragraphIdx, tokenIdx)
#   - in order to determine the class, count entity mentions between
#     the quote boundary and the entity span containing the quote head
#   - print a warning if there is an entity mention within a quote?
# - print the results

# left to do:
# - add the speaker offset classes for quotes
# - (also elsewhere: improve+commit WebAnno coreference processing)
# - combine annotations?
# ENTITY annotation -- add nesting level?
# 
# is:
# 3-6776914,15,27,15,28,*->3-2|*->4-3,*[3]|*
# 3-6776914,15,29,15,31,*->3-2,
#
# should be:
# 3-6776914,15,27,15,28,*->4-3,
# 3-6776914,15,27,15,31,*->3-2,
# 
# 3-6776914 11 15 27 Pirkanmaan O BB-ENTITY  (nesting level: 2)
# 3-6776914 11 15 28 Jätehuollon O II-ENTITY
# 3-6776914 11 15 29 toimitusjohtaja O I-ENTITY
# 3-6776914 11 15 30 Harri O I-ENTITY
# 3-6776914 11 15 31 Kallio O I-ENTITY
# 3-6776914 11 15 32 sanoo O O
# 3-6776914 11 15 33 . O O

# TODO two classification heads:
# https://discuss.huggingface.co/t/fine-tune-bert-with-two-classification-heads-next-to-each-other/9984
# TODO bug: don't add 1 to the class if authorHead points to an entity
# TODO bug: process nested coreference annotations correctly (how?)
# - do they always have the same structure? how many levels max.?
# - first: discard nested coreferences?

QUOTE_LABELS = ['O',
          'B-DIRECT+1', 'I-DIRECT+1', 'B-INDIRECT+1', 'I-INDIRECT+1',
          'B-DIRECT+2', 'I-DIRECT+2', 'B-INDIRECT+2', 'I-INDIRECT+2',
          'B-DIRECT+3', 'I-DIRECT+3', 'B-INDIRECT+3', 'I-INDIRECT+3',
          'B-DIRECT00', 'I-DIRECT00', 'B-INDIRECT00', 'I-INDIRECT00',
          'B-DIRECT=1', 'I-DIRECT=1', 'B-INDIRECT=1', 'I-INDIRECT=1',
          'B-DIRECT=2', 'I-DIRECT=2', 'B-INDIRECT=2', 'I-INDIRECT=2',
          'B-DIRECT=3', 'I-DIRECT=3', 'B-INDIRECT=3', 'I-INDIRECT=3',
          'B-DIRECT=4', 'I-DIRECT=4', 'B-INDIRECT=4', 'I-INDIRECT=4',
          'B-DIRECT-1', 'I-DIRECT-1', 'B-INDIRECT-1', 'I-INDIRECT-1',
          'B-DIRECT-2', 'I-DIRECT-2', 'B-INDIRECT-2', 'I-INDIRECT-2',
          'B-DIRECT-3', 'I-DIRECT-3', 'B-INDIRECT-3', 'I-INDIRECT-3']
ENT_LABELS = ['O', 'B-ENTITY', 'I-ENTITY']
LABELS = QUOTE_LABELS + [l for l in ENT_LABELS if l != 'O']
PAT_AUTHOR_HEAD = re.compile('([0-9]+)-([0-9]+)')


class TwoHeadedBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout \
                if config.classifier_dropout is not None \
                else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier_1 = torch.nn.Linear(config.hidden_size, len(QUOTE_LABELS))
        self.classifier_2 = torch.nn.Linear(config.hidden_size, len(ENT_LABELS))
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_1=None,
        labels_2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        seq_output = outputs[0]
        
        seq_output = self.dropout(seq_output)
        logits_1 = self.classifier_1(seq_output)
        logits_2 = self.classifier_2(seq_output)
        
        loss = None
        if labels_1 is not None and labels_2 is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # FIXME this ignores the attention mask
            #logits = torch.concat((logits_1, logits_2), dim=2)
            #print(logits.size())
            loss = loss_fct(logits_1.view(-1, len(QUOTE_LABELS)), labels_1.view(-1)) + \
                   loss_fct(logits_2.view(-1, len(ENT_LABELS)), labels_2.view(-1))
        
        if not return_dict:
            # FIXME this will fail
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return TokenClassifierOutput(
            loss=loss,
            logits = (logits_1, logits_2),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DataCollatorForTwoHeadedBert(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

        label_names = ['labels_1', 'labels_2']
        # TODO finished here
        # - what should the function return???
        # - a dictionary: { labels_1: <tensor>, labels_2: <tensor> }
        labels = { label_name: [feature[label_name] for feature in features] for label_name in label_names }
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            for label_name in label_names:
                batch[label_name] = [
                    label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels[label_name]
                ]
        else:
            for label_name in label_names:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels[label_name]
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

    def tf_call(self, features):
        raise NotImplementedError()

    def numpy_call(self, features):
        raise NotImplementedError()


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
    cl = Sequence(ClassLabel(num_classes=len(QUOTE_LABELS), names=QUOTE_LABELS))
    labels_per_doc = []
    for d in docs:
        labels = ['O' for t in d['tokens']]
        if d['doc_id'] in anns:
            tok_inv_dict = { (d['s_ids'][i], d['w_ids'][i]) : i \
                             for i in range(len(d['s_ids'])) }
            for a in anns[d['doc_id']]:
                author_idx = None
                m = PAT_AUTHOR_HEAD.match(a['QuoteAuthorLink.Lemma'])
                if m is not None:
                    author_idx = (int(m.group(1)), int(m.group(2)))
                start_idx = (int(a['startSentenceId']), int(a['startWordId']))
                end_idx = (int(a['endSentenceId']), int(a['endWordId']))
                author_cls = '00'
                i = None
                if start_idx in tok_inv_dict and end_idx in tok_inv_dict: # TODO marginal cases later
                    i = tok_inv_dict[start_idx]
                    j = tok_inv_dict[end_idx]
                    if author_idx is not None and author_idx in tok_inv_dict:
                        q = tok_inv_dict[author_idx]
                        if q < i:
                            author_cls = '-' + str(len([k for k in range(q+1, i) if d['entity-tags'][k] == 1])+1)
                        elif q < j:
                            author_cls = '=' + str(len([k for k in range(i, q) if d['entity-tags'][k] == 1])+ (0 if d['entity-tags'][q] == 2 else 1) )
                        else:
                            author_cls = '+' + str(len([k for k in range(j, q) if d['entity-tags'][k] == 1])+ (0 if d['entity-tags'][q] == 2 else 1) )
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
                labels[i] = 'B-{}{}'.format(lbl_direct, author_cls)
                i += 1
                while i < len(d['tokens']) \
                      and (d['s_ids'][i] < int(a['endSentenceId']) \
                           or (d['s_ids'][i] == int(a['endSentenceId']) \
                               and d['w_ids'][i] <= int(a['endWordId']))):
                    labels[i] = 'I-{}{}'.format(lbl_direct, author_cls)
                    i += 1
        labels_per_doc.append(labels)
    docs = docs.add_column('quote-tags', labels_per_doc)\
               .cast_column('quote-tags', cl)
    return docs

def add_entity_annotations(docs, anns):
    cl = Sequence(ClassLabel(num_classes=len(ENT_LABELS), names=ENT_LABELS))
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
                labels[i] = 'B-ENTITY'
                i += 1
                while i < len(d['tokens']) \
                      and (d['s_ids'][i] < int(a['endSentenceId']) \
                           or (d['s_ids'][i] == int(a['endSentenceId']) \
                               and d['w_ids'][i] <= int(a['endWordId']))):
                    labels[i] = 'I-ENTITY'
                    i += 1
        labels_per_doc.append(labels)
    docs = docs.add_column('entity-tags', labels_per_doc)\
               .cast_column('entity-tags', cl)
    return docs


def merge_annotations(docs):
    cl = Sequence(ClassLabel(num_classes=len(LABELS), names=LABELS))
    labels_per_doc = []
    for d in docs:
        labels_per_doc.append([
            ENT_LABELS[d['entity-tags'][i]] \
                if ENT_LABELS[d['entity-tags'][i]] != 'O' \
                else QUOTE_LABELS[d['quote-tags'][i]] \
            for i in range(len(d['entity-tags']))])
    docs = docs.add_column('merged-tags', labels_per_doc)\
               .cast_column('merged-tags', cl)
    return docs


#def tokenize_and_align_labels(examples, max_length):
#    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=max_length)
#    labels = []
#    for i, label in enumerate(examples['merged-tags']):
#        word_ids = tokenized_inputs.word_ids(batch_index=i)
#        previous_word_idx = None
#        label_ids = []
#        for word_idx in word_ids:  # Set the special tokens to -100.
#            if word_idx is None:
#                label_ids.append(-100)
#            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                label_ids.append(label[word_idx])
#            else:
#                label_ids.append(-100)
#            previous_word_idx = word_idx
#        labels.append(label_ids)
#    
#    tokenized_inputs['labels'] = labels
#    return tokenized_inputs


def tokenize_and_align_labels(examples, max_length):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True, is_split_into_words=True, max_length=max_length)
    labels_1, labels_2 = [], []
    for i, (label_1, label_2) in enumerate(zip(examples['quote-tags'], examples['entity-tags'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_1_ids, label_2_ids = [], []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_1_ids.append(-100)
                label_2_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_1_ids.append(label_1[word_idx])
                label_2_ids.append(label_2[word_idx])
            else:
                label_1_ids.append(-100)
                label_2_ids.append(-100)
            previous_word_idx = word_idx
        labels_1.append(label_1_ids)
        labels_2.append(label_2_ids)
    tokenized_inputs['labels_1'] = labels_1
    tokenized_inputs['labels_2'] = labels_2
    #tokenized_inputs['labels'] = labels_1
    return tokenized_inputs


def extract_spans(x):
    spans = []
    cur_start, cur_label, cur_cont = None, None, None
    for i in range(len(x)):
        # finish the current span
        if cur_label is not None and x[i] != 'I-'+cur_label:
            spans.append((cur_start, i, cur_label, cur_cont))
            cur_start, cur_label, cur_cont = None, None, None
        # start a new span
        if x[i].startswith('B-'):
            cur_start = i
            cur_label = x[i][2:]
            cur_cont = False
        # continue a quote span that was interrupted by a quote
        elif len(spans) >= 2 and x[i].startswith('I-') \
                and x[i][2:] == spans[-2][2] and i == spans[-1][1]+1:
            cur_start = spans[-2][0]
            cur_label = x[i][2:]
            cur_cont = True
    if cur_label is not None:    
        spans.append((cur_start, i, cur_label, cur_cont))
    return spans


def group_spans(spans):
    quotes = []
    for start, end, label, cont in spans:
        if label != 'ENTITY':
            if not cont:
                quotes.append({ 'label': label, 'start': start, 'end': end,
                                'ents_before': [], 'ents_inside': [], 'ents_after': [] })
            elif len(quotes) > 0:
                quotes[-1]['end'] = end
    for start, end, label, cont in spans:
        if label == 'ENTITY':
            for q in quotes:
                if end-1 < q['start']:
                    q['ents_before'].append((start, end))
                elif start > q['end']-1:
                    q['ents_after'].append((start, end))
                else:
                    q['ents_inside'].append((start, end))
    for q in quotes:
        q['author'] = None
        direction, offset = q['label'][-2], int(q['label'][-1])
        if direction == '0':
            continue
        entities = None
        if direction == '-':
            entities = q['ents_before']
        elif direction == '=':
            entities = q['ents_inside']
        elif direction == '+':
            entities = q['ents_after']
        if len(entities) >= offset:
            q['author'] = entities[-offset] if direction == '-' else entities[offset-1]
    return quotes


def parse_arguments():
    parser = argparse.ArgumentParser(description='Quote detection using BERT.')
    parser.add_argument('cmd', choices=['train', 'run', 'preprocess'])
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-m', '--model-dir', metavar='DIR')
    parser.add_argument('-a', '--annotations-file', metavar='FILE')
    parser.add_argument('-e', '--entities-file', metavar='FILE')
    parser.add_argument('-n', '--max-length', type=int, default=20,
                        help='maximum sequence length')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.cmd == 'preprocess':
        docs = load_dataset_from_conll(args.input_file)
        quote_anns = load_annotations(args.annotations_file)
        entity_anns = load_annotations(args.entities_file)
        docs = add_entity_annotations(docs, entity_anns)
        docs = add_quote_annotations(docs, quote_anns)
        #docs = merge_annotations(docs)

        for d in docs:
            for i, tok in enumerate(d['tokens']):
                print(d['doc_id'], d['par_id'], d['s_ids'][i], d['w_ids'][i], tok,
                      QUOTE_LABELS[d['quote-tags'][i]], ENT_LABELS[d['entity-tags'][i]])
            print()

    elif args.cmd == 'train':
        docs = load_dataset_from_conll(args.input_file)
        quote_anns = load_annotations(args.annotations_file)
        entity_anns = load_annotations(args.entities_file)
        docs = add_entity_annotations(docs, entity_anns)
        docs = add_quote_annotations(docs, quote_anns)
        #docs = merge_annotations(docs)
        #docs = docs.select(range(1000, 1100))

        tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
        model = TwoHeadedBert.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
        #model = AutoModelForTokenClassification.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1', num_labels=len(LABELS))
        
        tokenized_docs = docs.map(
            lambda x: tokenize_and_align_labels(x, 100),
            batched=True)
        
        data_collator = DataCollatorForTwoHeadedBert(
            tokenizer=tokenizer,
            padding='max_length',
            max_length = 100)
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
            tokenizer=tokenizer,
            data_collator=data_collator)
        trainer.train()
        
        model.save_pretrained(args.model_dir)

    elif args.cmd == 'run':
        
        docs = load_dataset_from_conll(args.input_file)
        tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')
        #model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        model = TwoHeadedBert.from_pretrained(args.model_dir)

        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=['articleId', 'startSentenceId', 'startWordId',
                        'endSentenceId', 'endWordId', 'direct', 'author', 'authorStart'])
        writer.writeheader()
        for d in tqdm.tqdm(docs):
            toks = tokenizer(d['tokens'], return_tensors='pt',
                             padding=True, is_split_into_words=True)
            pred = model(**toks)
            results_1 = torch.argmax(pred['logits'][0], 2).flatten()
            results_2 = torch.argmax(pred['logits'][1], 2).flatten()
            word_ids = toks.word_ids()
            # convert the result annotations from BERT tokens to words
            cur_word_id = None
            results_1_words, results_2_words = [], []
            for i in range(results_1.size()[0]):
                if word_ids[i] is None or word_ids[i] == cur_word_id:
                    continue
                cur_word_id = word_ids[i]
                print(d['tokens'][cur_word_id],
                      QUOTE_LABELS[results_1[i]],
                      ENT_LABELS[results_2[i]])
                results_1_words.append(QUOTE_LABELS[results_1[i]])
                results_2_words.append(ENT_LABELS[results_2[i]])
            #spans = extract_spans(results_words)
            #quotes = group_spans(spans)
            #for q in quotes:
            #    writer.writerow({
            #        'articleId': d['doc_id'], 
            #        'startSentenceId': d['s_ids'][q['start']],
            #        'startWordId': d['w_ids'][q['start']],
            #        'endSentenceId': d['s_ids'][q['end']-1], 
            #        'endWordId': d['w_ids'][q['end']-1], 
            #        'direct': str(q['label'].startswith('DIRECT')).lower(),
            #        'author': ' '.join(d['tokens'][i] for i in range(*q['author'])) \
            #                  if q['author'] is not None else '_',
            #        'authorStart': '{}-{}'.format(d['s_ids'][q['author'][0]],
            #                                      d['w_ids'][q['author'][0]]) \
            #                        if q['author'] is not None else '_',
            #    })


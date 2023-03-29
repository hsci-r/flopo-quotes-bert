import argparse
from collections import defaultdict
import csv

from flopo_formats.scripts.eval import load_corpus


def parse_author_head(x):
    return (int(x[:x.find('-')])-1, int(x[x.find('-')+1:])-1)

def compare_tokens(t_gs, t_out, doc_id, ents):
    if t_gs is None:
        return None
    if t_out is None:
        return 'unrecognized'
    if t_gs == t_out:
        return 'correct'
    if doc_id in ents and ents[doc_id][t_gs[0]][t_gs[1]] == ents[doc_id][t_out[0]][t_out[1]]:
        return 'correct'
    return 'incorrect'
    

def read_entities(filename, corpus):
    results = {}
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        cur_ent_id, cur_doc_id, cur_doc = 1, None, None
        for row in reader:
            if row['articleId'] != cur_doc_id:
                if cur_doc_id is not None:
                    results[cur_doc_id] = cur_doc
                cur_doc_id = row['articleId']
                cur_doc = [[None] * len(s) for s in corpus[cur_doc_id]]
                cur_ent_id = 1
            for i in range(int(row['startSentenceId'])-1, int(row['endSentenceId'])):
                for j in range(int(row['startWordId'])-1 if i == int(row['startSentenceId'])-1 else 0,
                               int(row['endWordId']) if i == int(row['endSentenceId'])-1 \
                                                       else len(cur_doc[i])):
                    cur_doc[i][j] = cur_ent_id
            cur_ent_id += 1
        results[cur_doc_id] = cur_doc
    return results

def read_quote_sources(filename, corpus):
    results = { doc_id: [[None] * len(s) for s in corpus[doc_id]] \
                for doc_id in corpus  }
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            author_field = 'authorHead' if 'authorHead' in row else 'authorStart'
            for i in range(int(row['startSentenceId'])-1, int(row['endSentenceId'])):
                for j in range(int(row['startWordId'])-1 if i == int(row['startSentenceId'])-1 else 0,
                               int(row['endWordId']) if i == int(row['endSentenceId'])-1 \
                                                       else len(results[row['articleId']][i])):
                    results[row['articleId']][i][j] = \
                        parse_author_head(row[author_field]) \
                        if '-' in row[author_field] else None
    return results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Quote detection using BERT.')
    parser.add_argument('-c', '--corpus-file', metavar='FILE')
    parser.add_argument('-a', '--annotations-file', metavar='FILE')
    parser.add_argument('-g', '--gs-file', metavar='FILE')
    parser.add_argument('-e', '--entities-file', metavar='FILE')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    corpus = load_corpus(args.corpus_file)
    ents = read_entities(args.entities_file, corpus)
    qs_gs = read_quote_sources(args.gs_file, corpus)
    qs_out = read_quote_sources(args.annotations_file, corpus)

    results = defaultdict(lambda: 0)
    for key in qs_gs:
        if args.verbose:
            print('===', key, '===')
            print()
        for s_gs, s_out in zip(qs_gs[key], qs_out[key]):
            test = [compare_tokens(t_gs, t_out, key, ents) \
                    for t_gs, t_out in zip(s_gs, s_out)]
            if not all([x is None for x in test]):
                if args.verbose:
                    print(s_gs)
                    print(s_out)
                    print(test)
                    print()
                for x in test:
                    if x is not None:
                        results[x] += 1
    
    print(sum(results.values()), results['correct']/sum(results.values()))
    for key, val in results.items():
        print(key, val)

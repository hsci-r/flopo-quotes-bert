# Quote detection with BERT

This repository contains the BERT-based quote detection model described
in the following publication:

Maciej Janicki, Antti Kanner and Eetu Mäkelä.
[Detection and attribution of quotes in Finnish news media: BERT vs. rule-based approach](https://openreview.net/forum?id=YTVwaoG0Mi).
In: *Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)*,
Tórshavn, Faroe Islands, May 2023.

## Usage

## Training a model

This puts a trained model into the directory `MODEL_DIR`:
```
python3 quotes-bert.py train -i INPUT_FILE -a ANNOTATIONS_FILE -e ENTITIES_FILE -m MODEL_DIR
```

The training requires the following input files:
- `INPUT_FILE` - a corpus in CoNLL-CSV format,
- `ANNOTATIONS_FILE` - a list of quotes in CSV format,
- `ENTITIES_FILE` - a list of coreference annotations in CSV format.

The repository
[hsci-r/fi-quote-coref-corpus](https://github.com/hsci-r/fi-quote-coref-corpus)
contains suitable training data for the model, in which the above three
files correspond to the `*-corpus.csv`, `*-quotes.csv` and `*-coref.csv` files.

## Running a trained model on new data

```
python3 quotes-bert.py run -i INPUT_FILE
```

This takes a corpus file as a parameter and returns a list of recognized
quotes.


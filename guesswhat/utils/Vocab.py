import os
import csv
import gzip
import json
from collections import Counter
from nltk.tokenize import TweetTokenizer


class Vocab():

    def __init__(self, file, min_occ):

        self.w2i, self.i2w, self.w2c = dict(), dict(), dict()

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                w, c = row[0], int(row[1])
                if c >= min_occ:
                    self.w2i[w] = len(self.w2i)
                    self.i2w[self.w2i[w]] = w
                    self.w2c[w] = c

        self.answer_tokens = [self['<yes>'], self['<no>'], self['<n/a>']]
        self.answers = {
            'yes': self['<yes>'],
            'no': self['<no>'],
            'n/a': self['<n/a>']}

    def __len__(self):
        return len(self.w2i)

    def __getitem__(self, q):

        if isinstance(q, str):
            return self.w2i.get(q, self.w2i['<unk>'])
        elif isinstance(q, int):
            return self.i2w.get(q, self.i2w[self.w2i['<unk>']])
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))

    def encode(self, x):
        return [self[xi] for xi in x]

    def encode_answer(self, x):
        return [self.answers[x]]

    def decode(self, x):
        return [self[xi] for xi in x]

    @classmethod
    def create(cls, data_dir, file, min_occ):

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eoq>', '<yes>', '<no>',
                          '<n/a>']

        w2c = Counter()
        tokenizer = TweetTokenizer(preserve_case=False)

        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                for qa in game['qas']:
                    words = tokenizer.tokenize(qa['question'])
                    w2c.update(words)

        out_file = os.path.join(data_dir, 'vocab.csv')
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

            for st in special_tokens:
                writer.writerow([st, 9999])

            for w, c in w2c.items():
                writer.writerow([w, c])

        return cls(out_file, min_occ)


class CategoryVocab():

    def __init__(self, file):

        self.c2i, self.i2c, self.c2count = dict(), dict(), dict()

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                category, count = row[0], int(row[1])
                self.c2i[category] = len(self.c2i)
                self.i2c[self.c2i[category]] = category
                self.c2count[category] = count

    def __len__(self):
        return len(self.c2i)

    def __getitem__(self, q):

        if isinstance(q, str):
            return self.c2i[q]
        elif isinstance(q, int):
            return self.i2c[q]
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))

    def encode(self, x):
        return [self.c2i[xi] for xi in x]

    def decode(self, x):
        return [self.i2c[xi] for xi in x]

    @classmethod
    def create(cls, file):

        special_tokens = ['<pad>']

        c2count = Counter()

        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                for oi, obj in enumerate(game['objects']):
                    c2count.update([obj['category']])

        out_file = '../data/categories.csv'
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

            for st in special_tokens:
                writer.writerow([st, 9999])

            for w, c in c2count.items():
                writer.writerow([w, c])

        return cls(out_file)

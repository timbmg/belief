import os
import gzip
import h5py
import json
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer


class QuestionerDataset(Dataset):

    def __init__(self, file, vocab, category_vocab, successful_only,
                 data_dir='data', unroll_dialogue=False):

        self.data = defaultdict(dict)

        features_file = os.path.join(data_dir, 'vgg_fc8.hdf5')
        mapping_file = os.path.join(data_dir, 'imagefile2id.json')
        for f in [features_file, mapping_file]:
            if not os.path.exists(f):
                raise FileNotFoundError("{} file not found." +
                                        "Please create with " +
                                        "utils/cache_visual_features.py"
                                        .format(features_file))

        self.features = np.asarray(h5py.File(features_file, 'r')['vgg_fc8'])
        self.mapping = json.load(open(mapping_file))

        tokenizer = TweetTokenizer(preserve_case=False)
        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if successful_only and game['status'] != 'success':
                    continue

                object_categories = list()
                object_bboxes = list()
                for oi, obj in enumerate(game['objects']):
                    object_categories.append(category_vocab[obj['category']])
                    object_bboxes.append(bb2feature(
                                            bbox=obj['bbox'],
                                            im_width=game['image']['width'],
                                            im_height=game['image']['height']))

                    if obj['id'] == game['object_id']:
                        target_id = oi

                source_dialogue = [vocab['<sos>']]
                target_dialogue = list()
                if unroll_dialogue:
                    source_questions = list()
                    target_questions = list()
                    question_lengths = list()
                    cumulative_lengths = list()
                    unrolled_dialogue = list()
                    previous_answer = [vocab['<sos>']]
                for qi, qa in enumerate(game['qas']):
                    question = tokenizer.tokenize(qa['question'])
                    qa = vocab.encode(question) \
                        + vocab.encode_answer(qa['answer'].lower())
                    source_dialogue += qa

                    if unroll_dialogue:
                        cumulative_lengths += [len(source_dialogue)]
                        question_lengths += [len(qa)]
                        unrolled_dialogue.append(deepcopy(source_dialogue))
                        source_questions.append(previous_answer +
                                                vocab.encode(question))
                        target_questions.append(vocab.encode(question) +
                                                [vocab['<eoq>']])

                        previous_answer = [qa[-1]]

                target_dialogue = [vocab['<eoq>'] if t in vocab.answer_tokens
                                   else t for t in source_dialogue[1:]]
                target_dialogue = target_dialogue + [vocab['<pad>']]

                image = game['image']['file_name']
                image_featuers = self.features[self.mapping[image]]

                idx = len(self.data)
                self.data[idx]['source_dialogue'] = source_dialogue
                self.data[idx]['target_dialogue'] = target_dialogue
                self.data[idx]['dialogue_lengths'] = len(source_dialogue)
                self.data[idx]['object_categories'] = object_categories
                self.data[idx]['object_bboxes'] = object_bboxes
                self.data[idx]['target_id'] = target_id
                self.data[idx]['num_objects'] = len(object_categories)
                self.data[idx]['image_url'] = game['image']['flickr_url']
                self.data[idx]['image'] = image
                self.data[idx]['image_featuers'] = image_featuers
                if unroll_dialogue:
                    self.data[idx]['source_questions'] = source_questions
                    self.data[idx]['target_questions'] = target_questions
                    self.data[idx]['num_questions'] = len(game['qas'])
                    self.data[idx]['question_lengths'] = question_lengths
                    self.data[idx]['cumulative_lengths'] = cumulative_lengths
                    self.data[idx]['unrolled_dialogue'] = unrolled_dialogue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            max_dialogue_length = max([d['dialogue_lengths'] for d in data])
            max_num_objects = max([d['num_objects'] for d in data])
            if 'unrolled_dialogue' in data[0].keys():
                max_num_questions = max([d['num_questions'] for d in data])
                # NOTE: this should be the same as max_dialogue_length
                max_cumulative_length = max([l for d in data
                                             for l in d['cumulative_lengths']])
                max_question_length = max([l for d in data
                                           for l in d['question_lengths']])

            batch = defaultdict(list)
            for item in data:  # TODO: refactor to example
                for key in data[0].keys():
                    padded = deepcopy(item[key])
                    if key in ['source_dialogue', 'target_dialogue']:
                        padded.extend(
                            [0]*(max_dialogue_length
                                 - item['dialogue_lengths']))

                    elif key in ['cumulative_lengths', 'question_lengths']:
                        padded.extend(
                            [0]*(max_num_questions
                                 - len(item['cumulative_lengths'])))

                    elif key in ['unrolled_dialogue', 'source_questions',
                                 'target_questions']:
                        if key in ['unrolled_dialogue']:
                            pad_up_to = max_cumulative_length
                        elif key in ['source_questions', 'target_questions']:
                            pad_up_to = max_question_length
                        for i in range(len(padded)):
                                padded[i].extend([0]*(pad_up_to
                                                      - len(padded[i])))
                        # pad dialogue up to max_num_questions
                        padded.extend(
                            [[0]*pad_up_to]
                            * (max_num_questions - item['num_questions']))

                    elif key in ['object_categories']:
                        padded.extend(
                            [0]*(max_num_objects-item['num_objects']))

                    elif key in ['object_bboxes']:
                        padded.extend(
                            [[0]*8]*(max_num_objects-item['num_objects']))

                    batch[key].append(padded)

            for k in batch.keys():
                if k in ['image', 'image_url']:
                    pass
                else:
                    batch[k] = torch.Tensor(batch[k]).to(device)
                    if k in ['source_dialogue', 'target_dialogue',
                             'num_questions', 'cumulative_lengths',
                             'dialogue_lengths', 'object_categories',
                             'num_objects', 'target_id', 'unrolled_dialogue',
                             'question_lengths', 'source_questions',
                             'target_questions']:
                        batch[k] = batch[k].long()

            return batch

        return collate_fn


class OracleDataset(Dataset):

    def __init__(self, file, vocab, category_vocab, successful_only):

        self.data = defaultdict(dict)

        self.answer2class = {
            'yes': 0,
            'no': 1,
            'n/a': 2}

        tokenizer = TweetTokenizer(preserve_case=False)
        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if successful_only and game['status'] != 'success':
                    continue

                for oi, obj in enumerate(game['objects']):
                    if obj['id'] == game['object_id']:
                        target_id = oi
                        target_category = category_vocab[obj['category']]
                        target_bbox = bb2feature(
                                        bbox=obj['bbox'],
                                        im_width=game['image']['width'],
                                        im_height=game['image']['height'])

                for qa in game['qas']:
                    question = tokenizer.tokenize(qa['question'])
                    question = vocab.encode(question)
                    answer = self.answer2class[qa['answer'].lower()]

                    idx = len(self.data)
                    self.data[idx]['question'] = question
                    self.data[idx]['question_lengths'] = len(question)
                    self.data[idx]['target_answer'] = answer
                    self.data[idx]['target_id'] = target_id
                    self.data[idx]['target_category'] = target_category
                    self.data[idx]['target_bbox'] = target_bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):
            max_question_lengths = max([d['question_lengths'] for d in data])

            batch = defaultdict(list)
            for item in data:
                for key in data[0].keys():
                    padded = deepcopy(item[key])
                    if key in ['question']:
                        padded.extend([0]*(max_question_lengths
                                           - item['question_lengths']))

                    batch[key].append(padded)

            for k in batch.keys():
                batch[k] = torch.Tensor(batch[k]).to(device)
                if k in ['question', 'question_lengths',
                         'target_answer', 'target_id',
                         'target_category']:
                    batch[k] = batch[k].long()

            return batch

        return collate_fn


class InferenceDataset(Dataset):

    def __init__(self, file, vocab, category_vocab, data_dir='data'):
        self.data = defaultdict(dict)

        features_file = os.path.join(data_dir, 'vgg_fc8.hdf5')
        mapping_file = os.path.join(data_dir, 'imagefile2id.json')
        for f in [features_file, mapping_file]:
            if not os.path.exists(f):
                raise FileNotFoundError("{} file not found." +
                                        "Please create with " +
                                        "utils/cache_visual_features.py"
                                        .format(features_file))

        self.features = np.asarray(h5py.File(features_file, 'r')['vgg_fc8'])
        self.mapping = json.load(open(mapping_file))

        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                source_dialogue = [vocab['<sos>']]

                object_categories = list()
                object_bboxes = list()
                for oi, obj in enumerate(game['objects']):
                    object_categories.append(category_vocab[obj['category']])
                    object_bboxes.append(bb2feature(
                                            bbox=obj['bbox'],
                                            im_width=game['image']['width'],
                                            im_height=game['image']['height']))

                    if obj['id'] == game['object_id']:
                        target_id = oi
                        target_category = category_vocab[obj['category']]
                        target_bbox = bb2feature(
                                        bbox=obj['bbox'],
                                        im_width=game['image']['width'],
                                        im_height=game['image']['height'])

                num_objects = len(object_categories)

                image = game['image']['file_name']
                image_featuers = self.features[self.mapping[image]]

                idx = len(self.data)
                self.data[idx]['source_dialogue'] = source_dialogue
                self.data[idx]['object_categories'] = object_categories
                self.data[idx]['object_bboxes'] = object_bboxes
                self.data[idx]['num_objects'] = num_objects
                self.data[idx]['target_id'] = target_id
                self.data[idx]['target_category'] = target_category
                self.data[idx]['target_bbox'] = target_bbox
                self.data[idx]['num_objects'] = num_objects
                self.data[idx]['image'] = image
                self.data[idx]['image_url'] = game['image']['flickr_url']
                self.data[idx]['image_featuers'] = image_featuers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):
            max_num_objects = max([d['num_objects'] for d in data])

            batch = defaultdict(list)
            for item in data:  # TODO: refactor to example
                for key in data[0].keys():
                    padded = deepcopy(item[key])
                    if key in ['object_categories']:
                        padded.extend(
                            [0]*(max_num_objects-item['num_objects']))

                    elif key in ['object_bboxes']:
                        padded.extend(
                            [[0]*8]*(max_num_objects-item['num_objects']))

                    batch[key].append(padded)

            for k in batch.keys():
                if k in ['image', 'image_url']:
                    pass
                else:
                    batch[k] = torch.Tensor(batch[k]).to(device)
                    if k in ['source_dialogue', 'object_categories',
                             'num_objects', 'target_id', 'target_category']:
                        batch[k] = batch[k].long()

            return batch

        return collate_fn


def bb2feature(bbox, im_width, im_height):
    x_width = bbox[2]
    y_height = bbox[3]

    x_left = bbox[0]
    y_upper = bbox[1]
    x_right = x_left+x_width
    y_lower = y_upper+y_height

    x_center = x_left + 0.5*x_width
    y_center = y_upper + 0.5*y_height

    # Rescale features fom -1 to 1
    x_left = (1.*x_left / im_width) * 2 - 1
    x_right = (1.*x_right / im_width) * 2 - 1
    x_center = (1.*x_center / im_width) * 2 - 1

    y_lower = (1.*y_lower / im_height) * 2 - 1
    y_upper = (1.*y_upper / im_height) * 2 - 1
    y_center = (1.*y_center / im_height) * 2 - 1

    x_width = (1.*x_width / im_width) * 2
    y_height = (1.*y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_upper, x_right, y_lower,
            x_center, y_center, x_width, y_height]

    return feat

import os
import gzip
import h5py
import json
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter, OrderedDict


class QuestionerDataset(Dataset):

    def __init__(self, file, vocab, category_vocab, successful_only,
                 data_dir='data', cumulative_dialogue=False,
                 mrcnn_objects=False, mrcnn_settings=None):

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

        self.mrcnn_objects = mrcnn_objects
        if self.mrcnn_objects:
            mrcnn_features_file = os.path.join(data_dir, 'mrcnn.hdf5')
            self.mrcnn_features = h5py.File(mrcnn_features_file, 'r')

            self.mrcnn_bboxes = np.asarray(self.mrcnn_features['boxes'])
            self.mrcnn_cats = np.asarray(self.mrcnn_features['class_probs'])

            mrcnn_mappig_file = os.path.join(
                data_dir, 'mrcnn_imagefile2id.json')
            self.mrcnn_mapping = json.load(open(mrcnn_mappig_file))
            self.filter_category = mrcnn_settings['filter_category']
            # self.remove_boxes = mrcnn_settings['remove_boxes']

            self.mrcnn_category_vocab = \
                {i: a for a, i in enumerate(mrcnn_classes)}

        target_cat_counter = Counter()
        tokenizer = TweetTokenizer(preserve_case=False)
        with gzip.open(file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if successful_only and game['status'] != 'success':
                    continue

                object_categories = list()
                object_bboxes = list()
                for oi, obj in enumerate(game['objects']):
                    if not self.mrcnn_objects:
                        object_categories.append(
                            category_vocab[obj['category']])
                        object_bboxes.append(bb2feature(
                            bbox=obj['bbox'],
                            im_width=game['image']['width'],
                            im_height=game['image']['height']))

                    if obj['id'] == game['object_id']:
                        target_id = oi
                        target_category_str = obj['category']
                        target_spatial = bb2feature(
                            bbox=obj['bbox'],
                            im_width=game['image']['width'],
                            im_height=game['image']['height'])
                target_category = \
                    category_vocab[game['objects'][target_id]['category']]
                target_cat_counter.update([target_category])

                if self.mrcnn_objects:
                    mrcnn_data = self.get_mrcnn_data(
                        game['image'],
                        {'target_category': target_category_str,
                         'target_spatial': target_spatial})

                    if mrcnn_settings['skip_below_05']:
                        if mrcnn_data['best_iou_val'] < 0.5:
                            continue
                    # map mrcnn category back to category vocab
                    for ci, c in enumerate(mrcnn_data['object_categories']):
                        mrcnn_data['object_categories'][ci] = \
                            category_vocab[mrcnn_classes[c]]

                source_dialogue = [vocab['<sos>']]
                target_dialogue = list()
                if cumulative_dialogue:
                    cumulative_lengths = [1]
                    question_lengths = [1]
                    cumulative_dialogue = [deepcopy(source_dialogue)]
                for qi, qa in enumerate(game['qas']):
                    question = tokenizer.tokenize(qa['question'])
                    qa = vocab.encode(question) \
                        + vocab.encode_answer(qa['answer'].lower())
                    source_dialogue += qa

                    if cumulative_dialogue:
                        cumulative_lengths += [len(source_dialogue)]
                        question_lengths += [len(qa)]
                        cumulative_dialogue.append(deepcopy(source_dialogue))

                # remove answers from target dialogue
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
                self.data[idx]['target_category'] = target_category
                self.data[idx]['num_objects'] = len(object_categories)
                self.data[idx]['image_url'] = game['image']['flickr_url']
                self.data[idx]['image'] = image
                self.data[idx]['image_featuers'] = image_featuers
                if cumulative_dialogue:
                    # num_questions +1 to include <sos>
                    self.data[idx]['num_questions'] = len(game['qas']) + 1
                    self.data[idx]['cumulative_lengths'] = cumulative_lengths
                    self.data[idx]['cumulative_dialogue'] = cumulative_dialogue
                if self.mrcnn_objects:
                    for k in mrcnn_data.keys():
                        self.data[idx][k] = mrcnn_data[k]

        target_cat_counter = OrderedDict(sorted(target_cat_counter.items()))
        # add [0] for padding
        self.category_weights = [0] + [min(target_cat_counter.values()) / cnt
                                       for cnt in target_cat_counter.values()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_mrcnn_data(self, game_image, meta):
        mrcnn_map_id = self.mrcnn_mapping[str(game_image['id'])]

        # Read Visual Features
        mrcnn_visual_featues = np.asarray(
            self.mrcnn_features["box_features"][mrcnn_map_id]
            ).reshape(-1, 1024)

        # Read Spatial Features
        mrcnn_game_bboxes = \
            np.asarray(self.mrcnn_bboxes[mrcnn_map_id]).reshape(-1, 4)
        mrcnn_spatials_features = list()
        for i in range(mrcnn_game_bboxes.shape[0]):
            # convert from mrcnn (xy top left, xy bottom right)
            # to (xy top left, wh)
            bbox = [mrcnn_game_bboxes[i][0], mrcnn_game_bboxes[i][1],
                    mrcnn_game_bboxes[i][2] - mrcnn_game_bboxes[i][0],
                    mrcnn_game_bboxes[i][3] - mrcnn_game_bboxes[i][1]]

            mrcnn_spatials_features.append(bb2feature(
                bbox=bbox,
                im_width=game_image['width'],
                im_height=game_image['height']))

        mrcnn_spatials_features = np.asarray(mrcnn_spatials_features)

        # Read Object Categories
        mrcnn_soft_cats = self.mrcnn_cats[mrcnn_map_id].reshape(-1, 81)
        mrcnn_obj_cats = np.argmax(mrcnn_soft_cats[:, 1:], 1) + 1
        num_objects = mrcnn_soft_cats.shape[0]

        # Filter Targets by Category
        mapped_category = self.mrcnn_category_vocab[meta['target_category']]
        cat_match = mrcnn_obj_cats == mapped_category
        if self.filter_category and np.sum(cat_match) > 0:
            candidate_mask = np.tile(np.expand_dims(cat_match, 1),
                                     (1, 8)).reshape(-1, 8)
            mrcnn_target_candidates_spatials = \
                candidate_mask * mrcnn_spatials_features
        else:
            mrcnn_target_candidates_spatials = mrcnn_spatials_features

        # Compute Targets by IOU between meta and mrcnn predictions
        meta_target_spatial = torch.Tensor(meta['target_spatial'])
        mrcnn_bbox_predictions = torch.Tensor(mrcnn_target_candidates_spatials)
        meta_mrcnn_target_iou = compute_iou(
            predictions=mrcnn_bbox_predictions, target=meta_target_spatial)

        best_iou_val, best_iou_id = torch.max(meta_mrcnn_target_iou, 0)
        iou_above_05 = meta_mrcnn_target_iou > 0.5
        """
        if self.remove_boxes:  # only for when we remove confusing boxes
            candidate_boxes = (meta_mrcnn_target_iou > 0.5).long()
            candidate_boxes[best_iou_id] = 0
            keep_indices = (1 - candidate_boxes).nonzero().squeeze()

            mrcnn_spatials = [a for a in np.array(
                mrcnn_spatials)[keep_indices.numpy()]]
            mrcnn_objects = mrcnn_objects[keep_indices.numpy()]
            soft_categories = soft_categories[keep_indices.numpy()]

            # taking the position of the best bbox in the new (with removed bboxes) matrix
            best = torch.max(best == keep_indices, 0)[1]

        if np.sum(cat_match) == 0:
            best_iou_id = [-1]
        """

        return {
            # same keys will overwrite self.data[idx]
            'target_id': best_iou_id.item(),
            'object_categories': mrcnn_obj_cats.tolist(),
            'object_bboxes': mrcnn_spatials_features.tolist(),
            'num_objects': num_objects,

            'best_iou_val': best_iou_val.item(),
            'multi_target_mask': iou_above_05.tolist(),
            'multi_target_ious': meta_mrcnn_target_iou.tolist(),
            # 'object_categories_soft': mrcnn_soft_cats.tolist(),
            'mrcnn_visual_features': mrcnn_visual_featues.tolist()
        }

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            max_dialogue_length = max([d['dialogue_lengths'] for d in data])
            max_num_objects = max([d['num_objects'] for d in data])
            if 'cumulative_dialogue' in data[0].keys():
                max_num_questions = max([d['num_questions'] for d in data])
                # NOTE: this should be the same as max_dialogue_length
                max_cumulative_length = max([l for d in data
                                             for l in d['cumulative_lengths']])

            batch = defaultdict(list)
            for item in data:  # TODO: refactor to example
                for key in data[0].keys():
                    padded = deepcopy(item[key])
                    if key in ['source_dialogue', 'target_dialogue']:
                        padded.extend(
                            [0] * (max_dialogue_length
                                   - item['dialogue_lengths']))

                    elif key in ['cumulative_lengths']:
                        padded.extend(
                            [0] * (max_num_questions
                                   - len(item['cumulative_lengths'])))

                    elif key in ['cumulative_dialogue']:
                        for i in range(len(padded)):
                            padded[i].extend([0] * (max_cumulative_length
                                                    - len(padded[i])))
                        # pad dialogue up to max_num_questions
                        padded.extend(
                            [[0] * max_cumulative_length]
                            * (max_num_questions - item['num_questions']))

                    elif key in ['object_categories', 'multi_target_mask',
                                 'multi_target_ious']:
                        padded.extend(
                            [0] * (max_num_objects - item['num_objects']))

                    elif key in ['object_bboxes']:
                        padded.extend(
                            [[0] * 8] * (max_num_objects
                                         - item['num_objects']))

                    elif key in ['mrcnn_visual_features']:
                        padded.extend(
                            [[0] * 1024] * (max_num_objects
                                            - item['num_objects']))

                    batch[key].append(padded)

            for k in batch.keys():
                if k in ['image', 'image_url']:
                    pass
                else:
                    batch[k] = torch.Tensor(batch[k]).to(device)

                    if k in ['source_dialogue', 'target_dialogue',
                             'num_questions', 'cumulative_lengths',
                             'dialogue_lengths', 'object_categories',
                             'num_objects', 'target_id', 'target_category',
                             'cumulative_dialogue', 'best_iou_val',
                             'multi_target_mask']:
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
                    self.data[idx]['num_objects'] = len(game['objects'])

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
                        padded.extend([0] * (max_question_lengths
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


class InferenceDataset(Dataset):  # TODO refactor

    def __init__(self, file, vocab, category_vocab, data_dir='data',
                 new_object=False):
        self.data = defaultdict(dict)
        self.new_object = new_object
        features_file = os.path.join(data_dir, 'vgg_fc8.hdf5')
        mapping_file = os.path.join(data_dir, 'imagefile2id.json')
        for f in [features_file, mapping_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(("{} file not found. Please create " +
                                         "with utils/cache_visual_features.py")
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
        # return 128
        return len(self.data)

    def __getitem__(self, idx):
        if not self.new_object:
            return self.data[idx]
        else:
            # sample a new object at random from available objects as target
            data = deepcopy(self.data[idx])
            random_object_id = np.random.randint(0, data['num_objects'])
            while random_object_id == data['target_id']:
                random_object_id = np.random.randint(0, data['num_objects'])
            data['target_id'] = random_object_id
            data['target_category'] = \
                data['object_categories'][data['target_id']]
            data['target_bbox'] = data['object_bboxes'][data['target_id']]
            return data

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
                            [0] * (max_num_objects - item['num_objects']))

                    elif key in ['object_bboxes']:
                        padded.extend(
                            [[0] * 8] * (max_num_objects - item['num_objects']))

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
    x_right = x_left + x_width
    y_lower = y_upper + y_height

    x_center = x_left + 0.5 * x_width
    y_center = y_upper + 0.5 * y_height

    # Rescale features fom -1 to 1
    x_left = (x_left / im_width) * 2 - 1
    x_right = (x_right / im_width) * 2 - 1
    x_center = (x_center / im_width) * 2 - 1

    y_lower = (y_lower / im_height) * 2 - 1
    y_upper = (y_upper / im_height) * 2 - 1
    y_center = (y_center / im_height) * 2 - 1

    x_width = (x_width / im_width) * 2
    y_height = (y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_upper, x_right, y_lower,
            x_center, y_center, x_width, y_height]

    return feat


def compute_iou(predictions, target):

    targets = target.view(1, -1).expand(predictions.size(0), 8)

    def elementwise_max(t1, t2):
        return torch.cat((t1.unsqueeze(1), t2.unsqueeze(1)), dim=1).max(1)[0]

    def elementwise_min(t1, t2):
        return torch.cat((t1.unsqueeze(1), t2.unsqueeze(1)), dim=1).min(1)[0]

    if predictions.size(1) > 4:
        predictions = predictions[:, 4:]
    if targets.size(1) > 4:
        targets = targets[:, 4:]

    coord1 = predictions.clone()
    coord1[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
    coord1[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
    coord1[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
    coord1[:, 3] = predictions[:, 1] + predictions[:, 3] / 2
    coord2 = targets.clone()
    coord2[:, 0] = targets[:, 0] - targets[:, 2] / 2
    coord2[:, 1] = targets[:, 1] - targets[:, 3] / 2
    coord2[:, 2] = targets[:, 0] + targets[:, 2] / 2
    coord2[:, 3] = targets[:, 1] + targets[:, 3] / 2

    dx = elementwise_min(coord1[:, 2], coord2[:, 2]) \
        - elementwise_max(coord1[:, 0], coord2[:, 0])
    dy = elementwise_min(coord1[:, 3], coord2[:, 3]) \
        - elementwise_max(coord1[:, 1], coord2[:, 1])

    # inter_area = (xI2 - xI1) * (yI2 - yI1)
    fail_xs = dx < 0
    fail_ys = dy < 0
    if type(fail_xs) == torch.autograd.Variable:
        fail_xs = fail_xs.data
        fail_ys = fail_ys.data
    inter_area = dx * dy

    bbox1_area = (coord2[:, 2] - coord2[:, 0]) * (coord2[:, 3] - coord2[:, 1])
    bbox2_area = (coord1[:, 2] - coord1[:, 0]) * (coord1[:, 3] - coord1[:, 1])
    union_area = (bbox1_area + bbox2_area) - inter_area

    iou = inter_area / union_area

    iou[fail_xs == 1] = 0
    iou[fail_ys == 1] = 0

    return iou


mrcnn_classes = [
    '<pad>', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

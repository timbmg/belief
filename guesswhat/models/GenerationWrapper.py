import torch


class GenerationWrapper():

    def __init__(self, qgen, guesser, oracle):

        self.qgen = qgen
        self.guesser = guesser
        self.oracle = oracle

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, sample, vocab, strategy, max_num_questions,
                 device, belief_state=False, mrcnn_belief=False,
                 mrcnn_guesser=False, return_keys=['generations']):

        batch_size = sample['source_dialogue'].size(0)
        dialogue = sample['source_dialogue'].clone()
        dialogue_lengths = dialogue.new_ones(batch_size)
        visual_features = sample['image_featuers']
        input = dialogue.new_empty(batch_size)\
            .fill_(vocab['<sos>']).unsqueeze(1)

        belief_kwargs = dict()
        if belief_state:
            belief_kwargs['dialogue'] = dialogue
            belief_kwargs['dialogue_lengths'] = dialogue_lengths
            belief_kwargs['object_categories'] = \
                sample['object_categories']
            belief_kwargs['object_bboxes'] = sample['object_bboxes']
            belief_kwargs['num_objects'] = sample['num_objects']
        if mrcnn_belief:
            belief_kwargs['guesser_visual_features'] = \
                sample['mrcnn_visual_features']

        if mrcnn_guesser or (not mrcnn_guesser and not mrcnn_belief):
            object_bboxes = sample['object_bboxes']
            object_categories = sample['object_categories']
        elif mrcnn_belief and not mrcnn_guesser:
            object_bboxes = sample['gt_object_bboxes']
            object_categories = sample['gt_object_categories']

        questions_lengths, h, c, return_dict = self.qgen.inference(
            input=input,
            visual_features=visual_features,
            end_of_question_token=vocab['<eoq>'],
            hidden=None,
            strategy=strategy,
            return_keys=return_keys,
            **belief_kwargs)

        hidden_states = torch.Tensor().to(device)
        mask = torch.ByteTensor().to(device)
        log_probs = torch.Tensor().to(device)
        object_probs = torch.Tensor().to(device)

        for qi in range(1, max_num_questions+1):

            if 'object_probs' in return_keys:
                if self.qgen.__class__.__name__ == 'QGen':
                    return_dict['object_probs'] = torch.nn.functional.softmax(
                        self.guesser(
                            dialogue=dialogue,
                            dialogue_lengths=dialogue_lengths,
                            object_categories=object_categories,
                            object_bboxes=object_bboxes,
                            num_objects=sample['num_objects'],
                            visual_features=sample.get('mrcnn_visual_features',
                                                       None)),
                        dim=-1).unsqueeze(1)
                object_probs = torch.cat(
                    (object_probs, return_dict['object_probs']), dim=1)

            # add question to dialogue
            dialogue = append_to_padded_sequence(
                padded_sequence=dialogue,
                sequence_lengths=dialogue_lengths,
                appendix=return_dict['generations'],
                appendix_lengths=questions_lengths)
            dialogue_lengths += questions_lengths

            if 'hidden_states' in return_keys:
                hidden_states = torch.cat(
                    (hidden_states, return_dict['hidden_states']), dim=1)
            if 'mask' in return_keys:
                mask = torch.cat((mask, return_dict['mask']), dim=1)
            if 'log_probs' in return_keys:
                log_probs = torch.cat(
                    (log_probs, return_dict['log_probs']), dim=1)

            # get answers
            answer_logits = self.oracle.forward(
                question=return_dict['generations'],
                question_lengths=questions_lengths,
                object_categories=sample['target_category'],
                object_bboxes=sample.get('gt_target_bbox',
                                         sample['target_bbox'])
                )
            answers = answer_logits.topk(1)[1].long()
            answers = answer_class_to_token(answers, vocab.w2i)

            # add answers to dialogue
            dialogue = append_to_padded_sequence(
                padded_sequence=dialogue,
                sequence_lengths=dialogue_lengths,
                appendix=answers,
                appendix_lengths=answers.new_ones(answers.size(0)))
            dialogue_lengths += 1

            if qi == max_num_questions:
                # last forward does not have to be done
                break

            if belief_state:
                # update dialogue with new q/a pair
                belief_kwargs['dialogue'] = dialogue
                belief_kwargs['dialogue_lengths'] = dialogue_lengths

            # ask next question
            questions_lengths, h, c, return_dict = self.qgen.inference(
                input=answers,
                visual_features=visual_features,
                end_of_question_token=vocab.w2i['<eoq>'],
                hidden=(h, c),
                strategy=strategy,
                return_keys=return_keys,
                **belief_kwargs)

        object_logits = self.guesser(
            dialogue=dialogue,
            dialogue_lengths=dialogue_lengths,
            object_categories=object_categories,
            object_bboxes=object_bboxes,
            num_objects=sample['num_objects'],
            visual_features=sample.get('mrcnn_visual_features', None))

        return_dict = dict()
        return_dict['dialogue'] = dialogue
        return_dict['object_logits'] = object_logits
        if 'hidden_states' in return_keys:
            return_dict['hidden_states'] = hidden_states
        if 'mask' in return_keys:
            return_dict['mask'] = mask
        if 'log_probs' in return_keys:
            return_dict['log_probs'] = log_probs
        if 'object_probs' in return_keys:
            return_dict['object_probs'] = torch.cat(
                (object_probs, torch.nn.functional.softmax(
                    object_logits, dim=-1).unsqueeze(1)),
                dim=1)

        return return_dict


def append_to_padded_sequence(padded_sequence, sequence_lengths, appendix,
                              appendix_lengths):

    max_length = torch.max(sequence_lengths + appendix_lengths)
    sequence = padded_sequence.new_zeros((padded_sequence.size(0), max_length))

    for si in range(len(padded_sequence)):
        new_length = sequence_lengths[si].item() + appendix_lengths[si].item()
        sequence[si, :new_length] = torch.cat(
            (padded_sequence[si, :sequence_lengths[si]],
             appendix[si, :appendix_lengths[si]]))

    return sequence


def answer_class_to_token(answers, w2i):

    yes_mask = answers == 0
    no_mask = answers == 1
    na_mask = answers == 2

    answers.masked_fill_(yes_mask, w2i['<yes>'])
    answers.masked_fill_(no_mask, w2i['<no>'])
    answers.masked_fill_(na_mask, w2i['<n/a>'])

    return answers

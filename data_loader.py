import copy
import csv
import json
import logging
import os
# from imblearn.over_sampling import SVMSMOTE,SMOTE
# from imblearn.combine import SMOTEENN
from collections import defaultdict
import torch
import collections
from torch.utils.data import TensorDataset, ConcatDataset
import statistics
from utils import get_label
import numpy as np
logger = logging.getLogger(__name__)

with open("indicator_list.json", 'r') as file:
    indicators = json.load(file)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        # self.indicator_feature = indicator_feature

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def write_answer(cls, lst, args):
        path = os.path.join(args.eval_dir, "answer_keys.txt")
        with open(path, 'w') as f:
            f.writelines(lst)

    def _create_examples(self, lines, set_type, args):
        """Creates examples for the training and dev sets."""
        if set_type == "test":
            lst = []
            for (i, line) in enumerate(lines):
                lst.append(str(i)+"\t"+line[0]+"\n")
            self.write_answer(lst, args)

        label_distribution = defaultdict(int)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            label_distribution[label] += 1
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        
        # simple over sampling
        # if set_type == "train":
        #     examples_cp = examples.copy()
        #     mean_ = int(statistics.mean(label_distribution.values()))
        #     for label, num in label_distribution.items():
        #         if num <= 330:
        #             power = int(mean_ / num) - 1
        #             if power == 0:
        #                 power = 1
        #             for each_sample in examples_cp:
        #                 if each_sample.label == label:
        #                     examples.extend([each_sample]*power)
        #             power = None

        return examples

    def get_examples(self, mode, args):
        """
        Args:
            mode: train, dev, get_examples
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "test":
            file_to_read = self.args.test_file
        elif mode == "pretrain":
            file_to_read = self.args.augmented_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), set_type=mode, args=args)


processors = {"semeval": SemEvalProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
):  
    num = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        # features list
        # lowercase = example.text_a.lower()
        # indicator_feature = []
        # for each_indicator in indicators:
        #     if each_indicator in lowercase:
        #         indicator_feature.append(1)
        #     else:
        #         indicator_feature.append(0)
        # split the large list into smaller list of size 384
        # assert len(indicator_feature) <= 2 * max_seq_len
        
        # indicator_feature = indicator_feature + [0] * (768 - len(indicator_feature))
        # print(indicator_feature)

        tokens_a = tokenizer.tokenize(example.text_a)

        e11_p = tokens_a.index("<s>")  # the start position of entity1
        e12_p = tokens_a.index("</s>")  # the end position of entity1
        e21_p = tokens_a.index("<o>")  # the start position of entity2
        e22_p = tokens_a.index("</o>")  # the end position of entity2

        # Replace the token
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count:
            print("here!")
            num += 1
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
   
        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            # logger.info("indicator_feature: %s" % " ".join([str(x) for x in indicator_feature]))
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
                # indicator_feature = indicator_feature
            )
        )
    print(num)
    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train", args)
        elif mode == "pretrain":
            examples = processor.get_examples("pretrain", args)
        elif mode == "test":
            examples = processor.get_examples("test", args)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # all_indicator_feature = torch.tensor([f.indicator_feature for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_e1_mask,
        all_e2_mask,
        # all_indicator_feature
    )

    return dataset


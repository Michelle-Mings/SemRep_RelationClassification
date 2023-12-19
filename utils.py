import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys

ADDITIONAL_SPECIAL_TOKENS = ["<o>", "</o>", "<s>", "</s>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, relation_labels[pred]))

def init_logger(logfile):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.INFO
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels, args):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def acc_and_f1(preds, labels):
  return {
      "f1_macro": f1_score(labels, preds, average='macro'),
      "f1_micro": f1_score(labels, preds, average='micro'),
      "f1_weighted": f1_score(labels, preds, average='weighted'),
      "acc": accuracy_score(labels, preds)
  }

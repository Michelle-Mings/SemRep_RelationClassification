import argparse
import logging
import os
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from model import RBERT
from utils import get_label, init_logger, load_tokenizer

logger = logging.getLogger(__name__)

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    # Check whether model exists
   
    print(pred_config.model_dir)
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")
    
    model = RBERT.from_pretrained(pred_config.model_dir, args=args, num_samples = 2, output_attentions=True)
    try:
        model = RBERT.from_pretrained(pred_config.model_dir, args=args, num_samples = 2, output_attentions=True)
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def convert_input_file_to_tensor_dataset(
    pred_config,
    args,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_e1_mask = []
    all_e2_mask = []
    total_line = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        for each_line in tqdm(f, total=total_lines, desc="Processing lines"):
            line = each_line.split("\t")[1]
            
            line = line.strip()
            tokens = tokenizer.tokenize(line)

            e11_p = tokens.index("<s>")  # the start position of entity1
            e12_p = tokens.index("</s>")  # the end position of entity1
            e21_p = tokens.index("<o>")  # the start position of entity2
            e22_p = tokens.index("</o>")  # the end position of entity2

            # Replace the token
            tokens[e11_p] = "$"
            tokens[e12_p] = "$"
            tokens[e21_p] = "#"
            tokens[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # assert for entity start index and end index 
            assert e11_p <= e12_p
            assert e21_p <= e22_p

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if args.add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            if args.add_sep_token:
                tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_e1_mask.append(e1_mask)
            all_e2_mask.append(e2_mask)
            total_line.append(each_line)
    
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_e1_mask = torch.tensor(all_e1_mask, dtype=torch.long)
    all_e2_mask = torch.tensor(all_e2_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask)

    return dataset, total_line


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    args.data_dir = "/ocean/projects/cis230035p/sming/con2/EntityTypeData/"
    model = load_model(pred_config, args, device)
    logger.info(args)

    dataset, write_to_output_file = convert_input_file_to_tensor_dataset(pred_config, args)
    
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": None,
                "e1_mask": batch[3],
                "e2_mask": batch[4],
            }
            outputs = model.predict(**inputs, output_attentions=True)
            
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    
    # Calculate softmax values for each sample within each batch
    softmax_values = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
    
    preds_ = np.argmax(preds, axis=1)
    
    max_values = np.max(softmax_values, axis=1)
    
    # Write to output file
    label_lst = get_label(args)
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for ori_input, max_val, pred_idx, one_prob in zip(write_to_output_file, max_values, preds_, softmax_values):
            
            add = [round(float(item),2) for item in one_prob]
            predicted_label = label_lst[pred_idx]
            threshold = LABEL_PROB[predicted_label]
            predict_prob = round(float(max_val),2)

            f.write("{}\t{}\t{}\t{}\n".format(ori_input.strip(),predicted_label, predict_prob, add))
                
    logger.info("Prediction Done!")


if __name__ == "__main__":
    # init_logger()
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file for prediction",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for prediction",
    )
    
    parser.add_argument("--model_dir", type=str, help="Path to save, load model")
    DO_PRETRAIN = False 

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    pred_config.do_pretrain = getattr(pred_config, 'do_pretrain', False)
    pred_config.do_pretrain = DO_PRETRAIN
    predict(pred_config)

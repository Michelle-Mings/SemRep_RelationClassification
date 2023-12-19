import torch
import torch.nn as nn
import logging
import argparse
from transformers import BertModel, BertPreTrainedModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import Tensor
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Sampler
from typing import Optional, Tuple
import torch.nn.functional as F
from utils import compute_metrics, get_label
logger = logging.getLogger(__name__)
import os

class FCLayer_(nn.Module):
	def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
		super(FCLayer_, self).__init__()
		self.use_activation = use_activation
		self.dropout = nn.Dropout(dropout_rate)
		self.linear = nn.Linear(input_dim, output_dim)
		self.tanh = nn.Tanh()
		self.norm = nn.LayerNorm(output_dim)

	def forward(self, x):
		x = self.linear(x)
		if self.use_activation:
			x = self.tanh(x)
		x = self.dropout(x)
		x = self.norm(x)
		return x


class PreTrainedBERT(BertPreTrainedModel):
	def __init__(self, config, args, from_func):
		super(PreTrainedBERT, self).__init__(config)
		self.args = args
		self.bert = BertModel(config=config)
		logger.info("***** Pre-Trained BERT Base loaded! *****")
		self.num_labels = config.num_labels
		self.cls_fc_layer = FCLayer_(config.hidden_size, config.hidden_size, 0.1)
		

	@staticmethod
	def get_loss_pair(i, j):
		'''
		input: a pair of samples
		'''
		score = torch.exp(F.cosine_similarity(i, j, dim=1)/0.1)
		return score
  
	def infer(self, input_ids, attention_mask, token_type_ids):
		outputs = self.bert(
			input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
		) 
		attentions = outputs[2]
		sequence_output = outputs[0]
		pooled_output = outputs[1]
		if self.args.do_pretrain:
			pooled_output = self.cls_fc_layer(pooled_output)
		return (sequence_output, pooled_output, attentions)

	def forward(self, input_ids, attention_mask, token_type_ids, labels):
		'''
		the input training data is arranged in a way that two adjcent sentneces are orginally augmented from the same sentence, so they are positive pair,
		the input is a batch each time
		'''
		outputs = self.bert(
			input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
		)
		
		attentions = outputs[2]
		pooled_output = outputs[1]
		if self.args.do_pretrain:
			pooled_output = self.cls_fc_layer(pooled_output)

		sample_outputs = pooled_output.split(1, dim=0)
		# Loop over even samples
		batch_loss = 0
		for i in range(0, len(sample_outputs), 2):
			sample_even = sample_outputs[i]
			sample_odd = sample_outputs[i+1]
			positive_pair_loss = self.get_loss_pair(sample_even, sample_odd)
			sum_ = 0
			sum_ += positive_pair_loss
			for j in range(0, len(sample_outputs)):
				if j == i+1:
					continue
				if j == i:
					continue
				this_pair = self.get_loss_pair(sample_even,sample_outputs[j])
				sum_ += this_pair
			
			this_even_sample_loss = torch.log(sum_) - torch.log(positive_pair_loss)
			batch_loss += this_even_sample_loss
		
		# Loop over odd samples
		for i in range(1, len(sample_outputs), 2):
			sample_even = sample_outputs[i]
			sample_odd = sample_outputs[i-1] 
			positive_pair_loss = self.get_loss_pair(sample_even, sample_odd)
			sum_ = 0
			sum_ += positive_pair_loss
			for j in range(0, len(sample_outputs)):
				if j == i-1:
					continue
				if j == i:
					continue
				this_pair = self.get_loss_pair(sample_even,sample_outputs[j])
				sum_ += this_pair
			this_odd_sample_loss = torch.log(sum_) - torch.log(positive_pair_loss) #positive_pair_loss/sum_
			batch_loss += this_odd_sample_loss

		loss = batch_loss/self.args.pretrain_batch_size
		
		outputs = outputs[:1] + (pooled_output,) + (loss,) + (attentions, )  # return sequence_output, pooled_output, (loss), 
		
		return outputs


class PreTrainer(object):
	def __init__(self, args, pretrain_dataset=None, dev_dataset=None):
		self.args = args
		self.pretrain_dataset = pretrain_dataset
		self.dev_dataset = dev_dataset
		self.label_lst = get_label(args)
		self.num_labels = len(self.label_lst)

		self.config = BertConfig.from_pretrained(
			args.model_name_or_path,
			num_labels=self.num_labels,
			finetuning_task=args.task,
			id2label={str(i): label for i, label in enumerate(self.label_lst)},
			label2id={label: i for i, label in enumerate(self.label_lst)},
			output_attentions=True,
		)
		self.model = PreTrainedBERT.from_pretrained(args.model_name_or_path, config=self.config, args=args, from_func="pretrain.py")

		self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
		self.model.to(self.device)

	def train(self):
		
		pretrain_sampler = SequentialSampler(self.pretrain_dataset)
		train_dataloader = DataLoader(
			self.pretrain_dataset,
			sampler=pretrain_sampler,
			batch_size=self.args.pretrain_batch_size,
		)

		if self.args.max_steps > 0:
			t_total = self.args.max_steps
			self.args.num_pretrain_epochs = (
				self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
			)
		else:
			t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_pretrain_epochs

		# Prepare optimizer and schedule (linear warmup and decay)
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.args.weight_decay,
			},
			{
				"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]
		optimizer = AdamW(
			optimizer_grouped_parameters,
			lr=self.args.learning_rate,
			eps=self.args.adam_epsilon,
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.args.warmup_steps,
			num_training_steps=t_total,
		)

		# Train!
		logger.info("***** Running pretraining *****")
		logger.info("  Num examples = %d", len(self.pretrain_dataset))
		logger.info("  Num Epochs = %d", self.args.num_pretrain_epochs)
		logger.info("  Total train batch size = %d", self.args.pretrain_batch_size)
		logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
		logger.info("  Total optimization steps = %d", t_total)
		logger.info("  Logging steps = %d", self.args.logging_steps)
		logger.info("  Save steps = %d", self.args.save_steps)

		global_step = 0
		tr_loss = 0.0
		self.model.zero_grad()

		train_iterator = trange(int(self.args.num_pretrain_epochs), desc="Epoch")

		for _ in train_iterator:
			epoch_iterator = tqdm(train_dataloader, desc="Iteration")
			for step, batch in enumerate(epoch_iterator):
				self.model.train()
				batch = tuple(t.to(self.device) for t in batch)
				inputs = {
					"input_ids": batch[0],
					"attention_mask": batch[1],
					"token_type_ids": batch[2],
					"labels": batch[3],
				}
				
				outputs = self.model(**inputs)
				loss = outputs[2]
				if self.args.gradient_accumulation_steps > 1:
					loss = loss / self.args.gradient_accumulation_steps

				loss.backward()

				tr_loss += loss.item()
				if (step + 1) % self.args.gradient_accumulation_steps == 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

					optimizer.step()
					scheduler.step()
					self.model.zero_grad()
					global_step += 1

					if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
						self.save_model()

				if 0 < self.args.max_steps < global_step:
					epoch_iterator.close()
					break

			if 0 < self.args.max_steps < global_step:
				train_iterator.close()
				break
		
		
		self.save_model()
		return global_step, tr_loss / global_step
	
	def save_model(self):
		# Save model checkpoint (Overwrite)
		if not os.path.exists(self.args.pre_model_dir):
			os.makedirs(self.args.pre_model_dir)
		self.model.save_pretrained(self.args.pre_model_dir)
		logger.info("Saving model checkpoint to %s", self.args.pre_model_dir)
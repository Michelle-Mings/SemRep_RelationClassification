import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F
import os
from pretrain import PreTrainedBERT
import logging
logger = logging.getLogger(__name__)

def load_pretrained_model(args):
    # Check whether model exists
    print(args.pre_model_dir)

    if not os.path.exists(args.pre_model_dir):
        raise Exception("Model doesn't exists! Pre-Train first!")

    model = PreTrainedBERT.from_pretrained(args.pre_model_dir, args=args, from_func = "from model.py")
    logger.info("***** PreTrained Loaded *****")
    return model


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn
    
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
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

class RBERT(BertPreTrainedModel):

    def __init__(self, config, args, num_samples):
        super(RBERT, self).__init__(config)
        
        self.bert = load_pretrained_model(args)
        self.num_labels = config.num_labels
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size * 3, config.num_labels, args.dropout_rate, use_activation=False)
        
        self.num_samples = num_samples
        self.attention = MultiHeadAttention()
        
        self.linear_projection = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
       
    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

    def get_batch_class_weights(self, batch_labels):
        class_counts = torch.zeros(self.num_labels)
        for c in range(self.num_labels):
            class_counts[c] = (batch_labels == c).sum().item()
        class_weights = self.num_samples / (class_counts + 1e-8)
        return class_weights
    
    def predict(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, output_attentions = True):
        outputs = self.bert.infer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        attentions = outputs[2]
       
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        output, scores = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = self.cls_fc_layer(pooled_output)

        normalized_output = self.linear_projection(output)
        sentence_level_features = torch.mean(normalized_output, dim=1)
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        
        logits = self.label_classifier(concat_h)
        
        outputs = (logits,) +  outputs[:3]

        return outputs
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, output_attentions = True):
        outputs = self.predict(input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, output_attentions = True)
        logits = outputs[0]
        class_weight = self.get_batch_class_weights(labels).to("cuda")
        
        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=class_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))    
            outputs = (loss,) + outputs
        
        return outputs 

# coding=utf-8

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
try:
    from torchcrf import CRF
except Exception:
    from TorchCRF import CRF
from transformers import AutoConfig, AutoModel


class PhoBertForQuadABSA(nn.Module):
    def __init__(self, config, num_labels=2):
        super(PhoBertForQuadABSA, self).__init__()
        self.num_labels = [num_labels, 2]
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf_num = 6

        self.crf_batch_first = True
        try:
            self.crf = CRF(self.crf_num, batch_first=True)
        except TypeError:
            self.crf = CRF(self.crf_num)
            self.crf_batch_first = False
        self.dense_output = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.crf_num),
        )
        self.imp_asp_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels[1]),
        )
        self.imp_opi_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels[1]),
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=2, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = cls(config, num_labels=num_labels, **kwargs)
        model.bert = AutoModel.from_pretrained(model_name_or_path, config=config)
        return model

    def forward(self, aspect_input_ids, aspect_labels,
                aspect_token_type_ids, aspect_attention_mask,
                exist_imp_aspect, exist_imp_opinion):

        outputs = self.bert(
            input_ids=aspect_input_ids,
            token_type_ids=aspect_token_type_ids,
            attention_mask=aspect_attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = sequence_output[:, 0]

        loss_fct = CrossEntropyLoss()
        imp_aspect_exist = self.imp_asp_classifier(pooled_output)
        last_token_index = torch.sum(aspect_attention_mask, dim=-1) - 1
        imp_opinion_exist = self.imp_opi_classifier(sequence_output[range(sequence_output.shape[0]), last_token_index])

        imp_aspect_loss = loss_fct(imp_aspect_exist, exist_imp_aspect.view(-1))
        imp_opinion_loss = loss_fct(imp_opinion_exist, exist_imp_opinion.view(-1))

        max_seq_len = aspect_input_ids.size()[1]
        sequence_output = self.dense_output(sequence_output)
        sequence_output = sequence_output.view(-1, max_seq_len, self.crf_num)
        crf_mask = aspect_attention_mask.byte()
        crf_tags = aspect_labels
        crf_inputs = sequence_output
        if not self.crf_batch_first:
            crf_inputs = crf_inputs.transpose(0, 1)
            crf_tags = crf_tags.transpose(0, 1)
            crf_mask = crf_mask.transpose(0, 1)
        try:
            ae_ll = self.crf(crf_inputs, crf_tags, mask=crf_mask, reduction='mean')
        except TypeError:
            ae_ll = self.crf(crf_inputs, crf_tags, mask=crf_mask)
            if ae_ll.dim() > 0:
                ae_ll = ae_ll.mean()
        ae_loss = -ae_ll
        pred_tags = self.crf.decode(crf_inputs, mask=crf_mask)

        total_loss = ae_loss + imp_aspect_loss + imp_opinion_loss
        return [total_loss], [pred_tags, imp_aspect_exist, imp_opinion_exist]


class PhoBertForCategorySentiClassification(nn.Module):
    def __init__(self, config, num_labels=2):
        super(PhoBertForCategorySentiClassification, self).__init__()
        self.num_labels = [num_labels, 2]
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, num_labels),
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=2, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = cls(config, num_labels=num_labels, **kwargs)
        model.bert = AutoModel.from_pretrained(model_name_or_path, config=config)
        return model

    def forward(self, tokenizer, _e, aspect_input_ids,
                aspect_token_type_ids, aspect_attention_mask,
                candidate_aspect, candidate_opinion, label_id):

        aspect_seq_len = torch.max(torch.sum(aspect_attention_mask, dim=-1))
        max_seq_len = aspect_seq_len
        aspect_input_ids = aspect_input_ids[:, :max_seq_len].contiguous()
        aspect_token_type_ids = aspect_token_type_ids[:, :max_seq_len].contiguous()
        aspect_attention_mask = aspect_attention_mask[:, :max_seq_len].contiguous()
        candidate_aspect = candidate_aspect[:, :max_seq_len].contiguous()
        candidate_opinion = candidate_opinion[:, :max_seq_len].contiguous()

        outputs = self.bert(
            input_ids=aspect_input_ids,
            token_type_ids=aspect_token_type_ids,
            attention_mask=aspect_attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = sequence_output[:, 0]

        hidden_size = pooled_output.shape[-1]

        candidate_aspect_sum = torch.sum(candidate_aspect, -1).float()
        aspect_denominator = (candidate_aspect_sum + candidate_aspect_sum.eq(0).float()).unsqueeze(-1).repeat(1, hidden_size)
        candidate_aspect_rep = torch.div(torch.matmul(candidate_aspect.float().unsqueeze(1), sequence_output).squeeze(1), aspect_denominator)

        candidate_opinion_sum = torch.sum(candidate_opinion, -1).float()
        opinion_denominator = (candidate_opinion_sum + candidate_opinion_sum.eq(0).float()).unsqueeze(-1).repeat(1, hidden_size)
        candidate_opinion_rep = torch.div(torch.matmul(candidate_opinion.float().unsqueeze(1), sequence_output).squeeze(1), opinion_denominator)

        fused_feature = torch.cat([candidate_aspect_rep, candidate_opinion_rep], -1)
        fused_feature = self.classifier(self.dropout(fused_feature))
        cate_loss_fct = BCEWithLogitsLoss()
        loss = cate_loss_fct(fused_feature.view(-1, self.num_labels[0]), label_id.view(-1, self.num_labels[0]).float())
        return [loss], [fused_feature]

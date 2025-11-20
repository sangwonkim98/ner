# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TokenClassificationModel(nn.Module):
    """
    HuggingFace AutoModel 위에 NER용 classifier head 하나 얹은 형태.
    Dropout 비율 등은 config.yaml로부터 주입.
    """

    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()

        # 1) 모델 config 로드
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels
        )

        # 2) 본체 backbone 로드 (BERT, RoBERTa, ALBERT 등 자동 인식)
        self.model = AutoModel.from_pretrained(pretrained_model_name, config=self.config)

        # 3) dropout + classifier layer
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # AutoModel 은 일반적으로 last_hidden_state 반환
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs.last_hidden_state  # shape: (batch, seq_len, hidden)

        x = self.dropout(sequence_output)
        logits = self.classifier(x)  # shape: (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # CrossEntropyLoss expects (batch*seq, num_labels)
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": logits
        }
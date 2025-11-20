# src/dataset.py

from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from torch.utils.data import DataLoader


@dataclass
class NERDataModule:
    tokenizer: AutoTokenizer
    label_list: List[str]
    train_dataloader: DataLoader
    eval_dataloader: DataLoader
    test_dataloader: DataLoader


def load_conll2003_data():
    """
    슬라이드에서 준 코드 조각 그대로 사용.
    """
    dataset = load_dataset("conll2003")
    return dataset


def tokenize_and_align_labels(examples, tokenizer, max_length: int, label_all_tokens: bool):
    """
    - 예시 하나당 'tokens' (단어 리스트)와 'ner_tags' (정수 라벨 리스트)가 들어있음
    - subword 단위 토크나이즈 후, word_id 를 이용해 라벨을 토큰에 매핑
    - loss 계산에서 무시할 토큰은 label = -100 으로 설정
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    all_labels = examples["ner_tags"]
    new_labels: List[List[int]] = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_id = None
        label_ids: List[int] = []

        for word_id in word_ids:
            if word_id is None:
                # CLS, SEP 같은 special token 은 loss 에서 무시
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # 새 단어의 첫 subword
                label_ids.append(labels[word_id])
            else:
                # 같은 단어의 나머지 subword
                if label_all_tokens:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
            previous_word_id = word_id

        new_labels.append(label_ids)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def create_ner_dataloaders(config) -> NERDataModule:
    """
    config(dict 또는 OmegaConf) 를 받아서
    - tokenizer 로드
    - 데이터셋 로드 + 전처리
    - DataLoader 생성
    을 한 번에 해주는 함수.
    """
    model_name = config["model"]["pretrained_model_name"]
    max_length = config["data"]["max_length"]
    label_all_tokens = config["data"]["label_all_tokens"]
    train_subset_ratio = config["data"].get("train_subset_ratio", 1.0)

    # 1) tokenizer & raw dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_conll2003_data()

    # 라벨 이름 리스트 (O, B-PER, I-PER, ...)
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_list: List[str] = ner_feature.feature.names

    # 2) tokenization + label alignment
    def _tokenize_function(examples):
        return tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            max_length=max_length,
            label_all_tokens=label_all_tokens,
        )

    tokenized_datasets = raw_datasets.map(
        _tokenize_function,
        batched=True,
        remove_columns=["tokens", "pos_tags", "chunk_tags", "ner_tags", "id"],
    )

    # 3) subset (GPU 없으면 학습 데이터 일부만 사용)
    train_dataset = tokenized_datasets["train"]
    if train_subset_ratio < 1.0:
        limit = int(len(train_dataset) * train_subset_ratio)
        train_dataset = train_dataset.select(range(limit))

    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # 4) DataLoader
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    return NERDataModule(
        tokenizer=tokenizer,
        label_list=label_list,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
    )
# scripts/run_train.py
"""
CoNLL-2003 NER 학습/평가 실행 스크립트

역할:
1) config YAML 로드
2) 데이터 로더 준비 (src/dataset.py)
3) 모델 생성 (src/model.py)
4) Trainer로 학습 & 평가 (src/trainer.py)
"""

import argparse
import random
import os

import numpy as np
import torch
import yaml

from src.dataset import create_ner_dataloaders
from src.model import TokenClassificationModel
from src.trainer import Trainer  # 이건 trainer.py에서 직접 구현해야 함


# --------------------------
# utils: seed 고정
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# argument parsing
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.yaml",
        help="실험 설정이 담긴 YAML 파일 경로",
    )
    return parser.parse_args()


# --------------------------
# main
# --------------------------
def main():
    args = parse_args()

    # 1. config 로드
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # seed 고정
    set_seed(config["training"]["seed"])

    # 2. 데이터 로더 준비
    data_module = create_ner_dataloaders(config)
    label_list = data_module.label_list

    # 3. 모델 생성
    model = TokenClassificationModel(
        pretrained_model_name=config["model"]["pretrained_model_name"],
        num_labels=len(label_list),
        dropout=config["model"]["dropout"],
    )

    # 4. Trainer 준비
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
    )

    # 5. 학습 + 평가
    trainer.train()          # train + validation loop 안에서 F1 모니터링
    trainer.evaluate("test") # best 모델로 test set F1 측정 (Trainer 쪽에서 구현)


if __name__ == "__main__":
    main()
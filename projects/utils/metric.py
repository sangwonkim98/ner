import numpy as np
from seqeval.metrics import f1_score, classification_report

def compute_metrics(predictions, labels, label_list):
    # 예측값과 실제 라벨을 비교하여 F1 계산
    # Flatten 로직 필요
    return f1_score(true_predictions, true_labels)
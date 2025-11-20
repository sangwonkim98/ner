# src/trainer.py

import os
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter  # ✅ 텐서보드
from seqeval.metrics import classification_report, f1_score

from tqdm.auto import tqdm


class Trainer:
    """
    - train(): 학습 루프 (train + validation F1 측정)
    - evaluate(): 주어진 split(test/val)으로 F1 측정
    - TensorBoard에 loss/F1 기록
    """

    def __init__(self, model, data_module, config):
        self.model = model
        self.data = data_module
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # config 불러오기
        self.epochs = config["training"]["num_epochs"]
        self.lr = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.warmup_ratio = config["training"]["warmup_ratio"]
        self.output_dir = config["logging"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # ✅ TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))
        self.global_step = 0         # step 단위 scalar 기록용
        self.current_epoch = 0       # epoch 단위 F1 기록용

        # 옵티마이저
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # 스케줄러 준비
        total_steps = self.epochs * len(self.data.train_dataloader)
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.best_f1 = 0.0

    # ------------------------------------------------------
    # Train Loop
    # ------------------------------------------------------
    def train(self):
        print(f"🔥 Training started for {self.epochs} epochs")
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            print(f"\n===== Epoch {epoch}/{self.epochs} =====")
            self.model.train()

            total_loss = 0

            for batch in tqdm(self.data.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs["loss"]
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["max_grad_norm"]
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # ✅ step 단위 loss 기록
                self.writer.add_scalar(
                    "Loss/step", loss.item(), self.global_step
                )
                self.global_step += 1

            avg_loss = total_loss / len(self.data.train_dataloader)
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

            # ✅ epoch 단위 loss 기록
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)

            # -----------------------------
            # Validation step
            # -----------------------------
            val_f1 = self.evaluate("val")   # 내부에서 F1/val 기록

            # save best
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                save_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"⭐ Best model updated! F1={val_f1:.4f} saved to {save_path}")

        # 학습 끝나면 flush
        self.writer.flush()

    # ------------------------------------------------------
    # Evaluate (val or test)
    # ------------------------------------------------------
    def evaluate(self, split="val"):
        self.model.eval()

        if split == "val":
            dataloader = self.data.eval_dataloader
        else:
            dataloader = self.data.test_dataloader

        preds = []
        trues = []

        label_list = self.data.label_list

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                logits = outputs["logits"]  # (B, L, C)
                preds_batch = torch.argmax(logits, dim=-1).cpu().numpy()
                true_batch = batch["labels"].cpu().numpy()

                for p_seq, t_seq in zip(preds_batch, true_batch):
                    pred_tags = []
                    true_tags = []

                    for p, t in zip(p_seq, t_seq):
                        if t == -100:
                            continue  # ignore padding/subword
                        pred_tags.append(label_list[p])
                        true_tags.append(label_list[t])

                    preds.append(pred_tags)
                    trues.append(true_tags)

        f1 = f1_score(trues, preds)
        print(f"[{split.upper()}] F1-score: {f1:.4f}")

        # test일 경우 상세 리포트 출력
        if split == "test":
            print(classification_report(trues, preds))

        # ✅ TensorBoard에 F1 기록
        if split == "val":
            self.writer.add_scalar("F1/val", f1, self.current_epoch)
        elif split == "test":
            # test는 마지막 epoch 기준으로 기록
            self.writer.add_scalar("F1/test", f1, self.current_epoch)

        return f1
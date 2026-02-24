import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def coeff_determination(y_pred, y_true):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + 2.22044604925e-16)


class TorchLSTMNet(nn.Module):
    def __init__(self, state_len, hidden_size=64):
        super().__init__()
        self.l1 = nn.LSTM(input_size=state_len, hidden_size=hidden_size, batch_first=True)
        self.l2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, state_len)

    def forward(self, x):
        h1, _ = self.l1(x)
        h2, _ = self.l2(h1)
        return self.out(h2[:, -1, :])


class standard_lstm:
    def __init__(self, data, seq_num=8, model_tag="multistep"):
        np.random.seed(7)
        torch.manual_seed(7)

        self.device = torch.device("cpu")
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]
        self.model_tag = model_tag

        self.preproc_pipeline = Pipeline(
            [
                ("stdscaler", StandardScaler()),
                ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ]
        )
        self.data = self.preproc_pipeline.fit_transform(data)

        # Multi-timestep input window (L) for one-step autoregressive forecasting.
        self.seq_num = int(seq_num)
        # Pushforward rollout depth used during training (autoregressive supervision).
        self.pushforward_steps = 5
        self.total_size = np.shape(data)[0] - int(self.seq_num) - self.pushforward_steps + 1

        input_seq = np.zeros((self.total_size, self.seq_num, self.state_len), dtype=np.float32)
        output_seq = np.zeros((self.total_size, self.state_len), dtype=np.float32)
        rollout_targets = np.zeros(
            (self.total_size, self.pushforward_steps, self.state_len), dtype=np.float32
        )

        for t in range(self.total_size):
            input_seq[t, :, :] = self.data[t : t + self.seq_num, :]
            output_seq[t, :] = self.data[t + self.seq_num, :]
            rollout_targets[t, :, :] = self.data[
                t + self.seq_num : t + self.seq_num + self.pushforward_steps, :
            ]

        idx = np.arange(self.total_size)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]
        rollout_targets = rollout_targets[idx]

        split_test = int(0.9 * self.total_size)
        self.input_seq_test = input_seq[split_test:]
        self.output_seq_test = output_seq[split_test:]
        self.rollout_targets_test = rollout_targets[split_test:]
        input_seq = input_seq[:split_test]
        output_seq = output_seq[:split_test]
        rollout_targets = rollout_targets[:split_test]

        self.ntrain = int(0.8 * np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[: self.ntrain]
        self.output_seq_train = output_seq[: self.ntrain]
        self.rollout_targets_train = rollout_targets[: self.ntrain]
        self.input_seq_valid = input_seq[self.ntrain :]
        self.output_seq_valid = output_seq[self.ntrain :]
        self.rollout_targets_valid = rollout_targets[self.ntrain :]

        self.model = TorchLSTMNet(state_len=self.state_len, hidden_size=64).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1.0e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1.0e-5
        )

        self.train_loss_hist = []
        self.valid_loss_hist = []
        self.valid_rollout_rmse_hist = []

        self.ckpt_path = f"./checkpoints/my_checkpoint_{self.model_tag}.pt"
        os.makedirs("./checkpoints", exist_ok=True)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _predict_one_step(self, seq_batch):
        # seq_batch shape: [B, L, D] -> output shape: [B, D]
        return self.model(seq_batch)

    def _pushforward_loss(self, seed_seq, rollout_truth):
        # Closed-loop rollout loss: feed model predictions back into the input sequence.
        # seed_seq: [B, L, D], rollout_truth: [B, K, D]
        rollout_pred = []
        current = seed_seq
        for k in range(self.pushforward_steps):
            pred = self._predict_one_step(current)
            rollout_pred.append(pred)
            current = torch.cat([current[:, 1:, :], pred.unsqueeze(1)], dim=1)

        rollout_pred = torch.stack(rollout_pred, dim=1)
        return self.criterion(rollout_pred, rollout_truth)

    def train_model(self):
        stop_iter = 0
        patience = 12
        best_valid_loss = float("inf")

        self.num_batches = 20
        self.train_batch_size = max(1, int(self.ntrain / self.num_batches))
        self.valid_batch_size = max(1, int(self.nvalid / self.num_batches))

        max_epochs = 30
        for i in range(max_epochs):
            print(f"[{self.model_tag}] Training iteration:", i)
            self.model.train()

            train_loss_accum = 0.0
            train_batches = 0
            for batch in range(self.num_batches):
                s = batch * self.train_batch_size
                e = min((batch + 1) * self.train_batch_size, self.ntrain)
                if s >= e:
                    continue
                xb = self._to_tensor(self.input_seq_train[s:e])
                yb = self._to_tensor(self.output_seq_train[s:e])
                y_roll = self._to_tensor(self.rollout_targets_train[s:e])

                self.optimizer.zero_grad()
                pred = self._predict_one_step(xb)
                one_step_loss = self.criterion(pred, yb)
                pushforward_loss = self._pushforward_loss(xb, y_roll)
                loss = one_step_loss + 0.7 * pushforward_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss_accum += float(loss.item())
                train_batches += 1

            train_epoch_loss = train_loss_accum / max(1, train_batches)

            self.model.eval()
            valid_loss_accum = 0.0
            valid_batches = 0
            with torch.no_grad():
                for batch in range(self.num_batches):
                    s = batch * self.valid_batch_size
                    e = min((batch + 1) * self.valid_batch_size, self.nvalid)
                    if s >= e:
                        continue
                    xb = self._to_tensor(self.input_seq_valid[s:e])
                    yb = self._to_tensor(self.output_seq_valid[s:e])
                    y_roll = self._to_tensor(self.rollout_targets_valid[s:e])
                    pred = self._predict_one_step(xb)
                    one_step_loss = self.criterion(pred, yb)
                    pushforward_loss = self._pushforward_loss(xb, y_roll)
                    valid_loss_accum += float((one_step_loss + 0.7 * pushforward_loss).item())
                    valid_batches += 1

                valid_loss = valid_loss_accum / max(1, valid_batches)
                full_valid_pred = self._predict_one_step(self._to_tensor(self.input_seq_valid)).cpu().numpy()

            valid_r2 = coeff_determination(full_valid_pred, self.output_seq_valid)
            valid_rollout_rmse = self._evaluate_rollout_rmse(
                self.input_seq_valid, self.rollout_targets_valid
            )
            self.train_loss_hist.append(train_epoch_loss)
            self.valid_loss_hist.append(valid_loss)
            self.valid_rollout_rmse_hist.append(valid_rollout_rmse)
            self.scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                print(f"[{self.model_tag}] Improved validation loss from:", best_valid_loss, " to:", valid_loss)
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                print(f"[{self.model_tag}] Validation rollout RMSE:", valid_rollout_rmse)
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.ckpt_path)
                stop_iter = 0
            else:
                print(f"[{self.model_tag}] Validation loss (no improvement):", valid_loss)
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                print(f"[{self.model_tag}] Validation rollout RMSE:", valid_rollout_rmse)
                stop_iter += 1

            if stop_iter == patience:
                break

        self.model.eval()
        with torch.no_grad():
            test_pred = self._predict_one_step(self._to_tensor(self.input_seq_test)).cpu().numpy()
        test_loss = np.mean(np.square(test_pred - self.output_seq_test))
        test_r2 = coeff_determination(test_pred, self.output_seq_test)
        test_rollout_rmse = self._evaluate_rollout_rmse(
            self.input_seq_test, self.rollout_targets_test
        )
        print(f"[{self.model_tag}] Test loss:", test_loss)
        print(f"[{self.model_tag}] Test R2:", test_r2)
        print(f"[{self.model_tag}] Test rollout RMSE:", test_rollout_rmse)

        self._plot_training_history()
        self._plot_torch_lstm_schematic()

    def restore_model(self):
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval()
        print(f"[{self.model_tag}] Model restored successfully!")

    def model_inference(self, test_data):
        self.restore_model()

        test_data = self.preproc_pipeline.transform(test_data)
        test_total_size = np.shape(test_data)[0] - int(self.seq_num)

        rec_input_seq = test_data[: self.seq_num, :].reshape(1, self.seq_num, self.state_len).astype(np.float32)

        rec_output_seq = np.zeros((test_total_size, self.state_len), dtype=np.float32)
        for t in range(test_total_size):
            rec_output_seq[t, :] = test_data[t + self.seq_num, :]

        print(f"[{self.model_tag}] Making predictions on testing data")
        rec_pred = np.copy(rec_output_seq)
        for t in range(test_total_size):
            with torch.no_grad():
                pred = self._predict_one_step(self._to_tensor(rec_input_seq)).cpu().numpy()[0]
            rec_pred[t] = pred
            rec_input_seq[0, 0:-1, :] = rec_input_seq[0, 1:, :]
            rec_input_seq[0, -1, :] = rec_pred[t]

        rec_pred = self.preproc_pipeline.inverse_transform(rec_pred)
        rec_output_seq = self.preproc_pipeline.inverse_transform(rec_output_seq)

        for i in range(self.state_len):
            plt.figure()
            plt.title("Mode " + str(i))
            plt.plot(rec_pred[:, i], label="Predicted")
            plt.plot(rec_output_seq[:, i], label="True")
            plt.legend()
            plt.savefig("Mode_" + str(i) + f"_prediction_{self.model_tag}.png")
            plt.close()

        rollout_rmse = np.sqrt(np.mean(np.square(rec_pred - rec_output_seq), axis=0))
        rollout_mae = np.mean(np.abs(rec_pred - rec_output_seq), axis=0)
        print(f"[{self.model_tag}] Deployment RMSE per mode:", rollout_rmse)
        print(f"[{self.model_tag}] Deployment MAE per mode:", rollout_mae)
        print(f"[{self.model_tag}] Deployment RMSE (mean over modes):", np.mean(rollout_rmse))

        return rec_output_seq, rec_pred

    def _plot_training_history(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.train_loss_hist, marker="o", label="Train Loss")
        plt.plot(self.valid_loss_hist, marker="s", label="Validation Loss")
        if self.valid_rollout_rmse_hist:
            plt.plot(self.valid_rollout_rmse_hist, marker="^", label="Validation Rollout RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Torch LSTM Training History")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Training_Loss_{self.model_tag}.png")
        plt.close()

    def _plot_torch_lstm_schematic(self):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")

        boxes = [
            (0.03, 0.35, 0.18, 0.3, f"Input Sequence\n(L={self.seq_num}, r=3)"),
            (0.27, 0.35, 0.18, 0.3, "LSTM Layer 1\n(hidden=64)"),
            (0.51, 0.35, 0.18, 0.3, "LSTM Layer 2\n(hidden=64)"),
            (0.75, 0.35, 0.18, 0.3, "Linear Head\n(output=r)")
        ]

        for x, y, w, h, txt in boxes:
            rect = plt.Rectangle((x, y), w, h, facecolor="#d7efe5", edgecolor="#1f7a59", linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

        arrows = [
            (0.21, 0.5, 0.27, 0.5),
            (0.45, 0.5, 0.51, 0.5),
            (0.69, 0.5, 0.75, 0.5),
        ]
        for x1, y1, x2, y2 in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#1f2a24"))

        ax.text(
            0.5,
            0.08,
            "Pushforward training: K=5 closed-loop rollout loss + autoregressive deployment",
            ha="center",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(f"Torch_LSTM_Schematic_{self.model_tag}.png")
        plt.close()

    def _evaluate_rollout_rmse(self, input_seq, rollout_targets):
        self.model.eval()
        sqerr = []
        with torch.no_grad():
            for i in range(input_seq.shape[0]):
                current = self._to_tensor(input_seq[i : i + 1])
                preds = []
                for _ in range(self.pushforward_steps):
                    pred = self.model(current)
                    preds.append(pred.cpu().numpy()[0])
                    current = torch.cat([current[:, 1:, :], pred.unsqueeze(1)], dim=1)
                preds = np.stack(preds, axis=0)
                truth = rollout_targets[i]
                sqerr.append(np.mean(np.square(preds - truth)))
        return float(np.sqrt(np.mean(sqerr)))


if __name__ == "__main__":
    print("Torch architecture file")

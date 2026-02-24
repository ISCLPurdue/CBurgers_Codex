import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def coeff_determination(y_pred, y_true):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + 2.22044604925e-16)


class Seq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM for direct multi-step state rollout prediction."""

    def __init__(self, state_len, hidden_size=64):
        super().__init__()
        self.encoder = nn.LSTM(input_size=state_len, hidden_size=hidden_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=state_len, hidden_size=hidden_size)
        self.readout = nn.Linear(hidden_size, state_len)
        self.delta_scale = 0.05

    def forward_rollout(self, seed_seq, steps, tf_truth=None, tf_prob=0.0):
        # seed_seq: [B, L, D]
        _, (h, c) = self.encoder(seed_seq)
        h_t = h[-1]
        c_t = c[-1]

        # Decoder input is last observed state.
        dec_in = seed_seq[:, -1, :]
        preds = []
        for k in range(steps):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            # Bound predicted increment to prevent long-rollout blow-up.
            delta = torch.tanh(self.readout(h_t)) * self.delta_scale
            pred_abs = dec_in + delta
            preds.append(pred_abs)

            if tf_truth is not None and tf_prob > 0.0:
                mask = (torch.rand(pred_abs.shape[0], 1, device=pred_abs.device) < tf_prob).float()
                dec_in = mask * tf_truth[:, k, :] + (1.0 - mask) * pred_abs
            else:
                dec_in = pred_abs

        return torch.stack(preds, dim=1)


class standard_lstm:
    def __init__(self, data, seq_num=8, model_tag="multistep"):
        np.random.seed(7)
        torch.manual_seed(7)

        self.device = torch.device("cpu")
        self.state_len = np.shape(data)[1]
        self.model_tag = model_tag

        self.preproc_pipeline = Pipeline([("stdscaler", StandardScaler())])
        self.data = self.preproc_pipeline.fit_transform(data)

        self.mode_weights = torch.ones(self.state_len, dtype=torch.float32, device=self.device)

        self.seq_num = int(seq_num)
        self.max_pushforward_steps = 60
        self.total_size = np.shape(data)[0] - int(self.seq_num) - self.max_pushforward_steps + 1
        if self.total_size <= 0:
            raise ValueError("Not enough timesteps for chosen seq_num and rollout horizon")

        input_seq = np.zeros((self.total_size, self.seq_num, self.state_len), dtype=np.float32)
        output_seq = np.zeros((self.total_size, self.state_len), dtype=np.float32)
        rollout_targets = np.zeros(
            (self.total_size, self.max_pushforward_steps, self.state_len), dtype=np.float32
        )

        for t in range(self.total_size):
            input_seq[t, :, :] = self.data[t : t + self.seq_num, :]
            output_seq[t, :] = self.data[t + self.seq_num, :]
            rollout_targets[t, :, :] = self.data[
                t + self.seq_num : t + self.seq_num + self.max_pushforward_steps, :
            ]

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

        self.model = Seq2SeqLSTM(state_len=self.state_len, hidden_size=64).to(self.device)
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

    def _weighted_mse(self, pred, truth):
        diff = pred - truth
        return torch.mean((diff * self.mode_weights) ** 2)

    def _curriculum_steps(self, epoch, max_epochs):
        frac = float(epoch + 1) / float(max_epochs)
        target = int(8 + frac * (self.max_pushforward_steps - 8))
        return max(8, min(self.max_pushforward_steps, target))

    def _teacher_forcing_prob(self, epoch, max_epochs):
        frac = float(epoch) / float(max_epochs - 1)
        return 0.98 - 0.48 * frac

    def train_model(self):
        stop_iter = 0
        patience = 15
        best_valid_rollout_rmse = float("inf")

        num_batches = 20
        train_batch_size = max(1, int(self.ntrain / num_batches))
        valid_batch_size = max(1, int(self.nvalid / num_batches))

        max_epochs = 50
        for i in range(max_epochs):
            rollout_steps = self._curriculum_steps(i, max_epochs)
            tf_prob = self._teacher_forcing_prob(i, max_epochs)
            print(f"[{self.model_tag}] Training iteration:", i)
            print(f"[{self.model_tag}] Rollout steps={rollout_steps}, teacher_forcing_prob={tf_prob:.3f}")
            self.model.train()

            train_loss_accum = 0.0
            train_batches = 0
            for batch in range(num_batches):
                s = batch * train_batch_size
                e = min((batch + 1) * train_batch_size, self.ntrain)
                if s >= e:
                    continue

                xb = self._to_tensor(self.input_seq_train[s:e])
                yb = self._to_tensor(self.output_seq_train[s:e])
                y_roll = self._to_tensor(self.rollout_targets_train[s:e, :rollout_steps, :])

                self.optimizer.zero_grad()
                pred_roll = self.model.forward_rollout(xb, steps=rollout_steps, tf_truth=y_roll, tf_prob=tf_prob)

                one_step_loss = self._weighted_mse(pred_roll[:, 0, :], yb)
                rollout_loss = self._weighted_mse(pred_roll, y_roll)
                loss = 0.2 * one_step_loss + 0.8 * rollout_loss

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
                for batch in range(num_batches):
                    s = batch * valid_batch_size
                    e = min((batch + 1) * valid_batch_size, self.nvalid)
                    if s >= e:
                        continue
                    xb = self._to_tensor(self.input_seq_valid[s:e])
                    yb = self._to_tensor(self.output_seq_valid[s:e])
                    y_roll = self._to_tensor(self.rollout_targets_valid[s:e, :rollout_steps, :])

                    pred_roll = self.model.forward_rollout(xb, steps=rollout_steps, tf_truth=None, tf_prob=0.0)
                    one_step_loss = self._weighted_mse(pred_roll[:, 0, :], yb)
                    rollout_loss = self._weighted_mse(pred_roll, y_roll)
                    valid_loss_accum += float((0.2 * one_step_loss + 0.8 * rollout_loss).item())
                    valid_batches += 1

                valid_loss = valid_loss_accum / max(1, valid_batches)
                full_valid_pred = self.model.forward_rollout(
                    self._to_tensor(self.input_seq_valid), steps=1, tf_truth=None, tf_prob=0.0
                )[:, 0, :].cpu().numpy()

            valid_r2 = coeff_determination(full_valid_pred, self.output_seq_valid)
            valid_rollout_rmse = self._evaluate_rollout_rmse(
                self.input_seq_valid, self.rollout_targets_valid, rollout_steps=self.max_pushforward_steps
            )
            self.train_loss_hist.append(train_epoch_loss)
            self.valid_loss_hist.append(valid_loss)
            self.valid_rollout_rmse_hist.append(valid_rollout_rmse)
            self.scheduler.step(valid_loss)

            if valid_rollout_rmse < best_valid_rollout_rmse:
                print(
                    f"[{self.model_tag}] Improved validation rollout RMSE from:",
                    best_valid_rollout_rmse,
                    " to:",
                    valid_rollout_rmse,
                )
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                print(f"[{self.model_tag}] Validation rollout RMSE:", valid_rollout_rmse)
                best_valid_rollout_rmse = valid_rollout_rmse
                torch.save(self.model.state_dict(), self.ckpt_path)
                stop_iter = 0
            else:
                print(f"[{self.model_tag}] Validation rollout RMSE (no improvement):", valid_rollout_rmse)
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                stop_iter += 1

            if stop_iter == patience:
                break

        self.model.eval()
        with torch.no_grad():
            test_pred = self.model.forward_rollout(
                self._to_tensor(self.input_seq_test), steps=1, tf_truth=None, tf_prob=0.0
            )[:, 0, :].cpu().numpy()
        test_loss = np.mean(np.square(test_pred - self.output_seq_test))
        test_r2 = coeff_determination(test_pred, self.output_seq_test)
        test_rollout_rmse = self._evaluate_rollout_rmse(
            self.input_seq_test, self.rollout_targets_test, rollout_steps=self.max_pushforward_steps
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
                pred_abs = self.model.forward_rollout(
                    self._to_tensor(rec_input_seq), steps=1, tf_truth=None, tf_prob=0.0
                )[0, 0, :].cpu().numpy()
            rec_pred[t] = pred_abs
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
        plt.ylabel("Loss")
        plt.title("Seq2Seq LSTM Training History")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Training_Loss_{self.model_tag}.png")
        plt.close()

    def _plot_torch_lstm_schematic(self):
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")

        boxes = [
            (0.02, 0.35, 0.17, 0.3, f"Input Sequence\\n(L={self.seq_num}, r=3)"),
            (0.24, 0.35, 0.17, 0.3, "Encoder LSTM\\n(hidden=64)"),
            (0.46, 0.35, 0.17, 0.3, "Decoder LSTMCell\\n(hidden=64)"),
            (0.68, 0.35, 0.14, 0.3, "Linear\\n(delta r)"),
            (0.85, 0.35, 0.13, 0.3, "Rollout\\nK steps"),
        ]

        for x, y, w, h, txt in boxes:
            rect = plt.Rectangle((x, y), w, h, facecolor="#d7efe5", edgecolor="#1f7a59", linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

        arrows = [(0.19, 0.5, 0.24, 0.5), (0.41, 0.5, 0.46, 0.5), (0.63, 0.5, 0.68, 0.5), (0.82, 0.5, 0.85, 0.5)]
        for x1, y1, x2, y2 in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#1f2a24"))

        ax.text(
            0.5,
            0.08,
            "Seq2Seq rollout training with scheduled sampling and curriculum horizon",
            ha="center",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(f"Torch_LSTM_Schematic_{self.model_tag}.png")
        plt.close()

    def _evaluate_rollout_rmse(self, input_seq, rollout_targets, rollout_steps):
        self.model.eval()
        rollout_steps = min(rollout_steps, rollout_targets.shape[1])
        sqerr = []
        with torch.no_grad():
            for i in range(input_seq.shape[0]):
                seed = self._to_tensor(input_seq[i : i + 1])
                preds = self.model.forward_rollout(seed, steps=rollout_steps, tf_truth=None, tf_prob=0.0)
                pred_np = preds[0].cpu().numpy()
                truth = rollout_targets[i, :rollout_steps, :]
                sqerr.append(np.mean(np.square(pred_np - truth)))
        return float(np.sqrt(np.mean(sqerr)))


if __name__ == "__main__":
    print("Seq2Seq Torch architecture file")


class KoopmanStateSpaceNet(nn.Module):
    """Neural state-space model with stable linear latent dynamics + nonlinear residual."""

    def __init__(self, state_len, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_len, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_len),
        )
        self.k_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.05)
        self.residual = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _stable_k(self):
        # Spectral normalization keeps linear latent operator near stable regime.
        spec = torch.linalg.matrix_norm(self.k_raw, ord=2)
        scale = torch.clamp(spec, min=1.0)
        return self.k_raw / scale

    def one_step(self, x):
        z = self.encoder(x)
        k = self._stable_k()
        z_next = torch.matmul(z, k.T) + 0.1 * torch.tanh(self.residual(z))
        x_next = self.decoder(z_next)
        return x_next

    def rollout(self, seed_seq, steps, tf_truth=None, tf_prob=0.0):
        # Use last state in seed sequence as current state.
        x_curr = seed_seq[:, -1, :]
        preds = []
        for k in range(steps):
            x_next = self.one_step(x_curr)
            preds.append(x_next)
            if tf_truth is not None and tf_prob > 0.0:
                mask = (torch.rand(x_next.shape[0], 1, device=x_next.device) < tf_prob).float()
                x_curr = mask * tf_truth[:, k, :] + (1.0 - mask) * x_next
            else:
                x_curr = x_next
        return torch.stack(preds, dim=1)


class koopman_ssm:
    def __init__(self, data, seq_num=8, model_tag="koopman"):
        np.random.seed(11)
        torch.manual_seed(11)

        self.device = torch.device("cpu")
        self.state_len = np.shape(data)[1]
        self.model_tag = model_tag
        self.seq_num = int(seq_num)
        self.max_pushforward_steps = 60

        self.preproc_pipeline = Pipeline([("stdscaler", StandardScaler())])
        self.data = self.preproc_pipeline.fit_transform(data)
        self.total_size = np.shape(data)[0] - self.seq_num - self.max_pushforward_steps + 1
        if self.total_size <= 0:
            raise ValueError("Not enough timesteps for Koopman training.")

        input_seq = np.zeros((self.total_size, self.seq_num, self.state_len), dtype=np.float32)
        output_seq = np.zeros((self.total_size, self.state_len), dtype=np.float32)
        rollout_targets = np.zeros(
            (self.total_size, self.max_pushforward_steps, self.state_len), dtype=np.float32
        )
        for t in range(self.total_size):
            input_seq[t, :, :] = self.data[t : t + self.seq_num, :]
            output_seq[t, :] = self.data[t + self.seq_num, :]
            rollout_targets[t, :, :] = self.data[
                t + self.seq_num : t + self.seq_num + self.max_pushforward_steps, :
            ]

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

        self.model = KoopmanStateSpaceNet(state_len=self.state_len, latent_dim=16, hidden_dim=64).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1.0e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1.0e-5
        )
        self.mode_weights = torch.ones(self.state_len, dtype=torch.float32, device=self.device)
        self.train_loss_hist = []
        self.valid_loss_hist = []
        self.valid_rollout_rmse_hist = []
        self.ckpt_path = f"./checkpoints/my_checkpoint_{self.model_tag}.pt"
        os.makedirs("./checkpoints", exist_ok=True)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _weighted_mse(self, pred, truth):
        return torch.mean(((pred - truth) * self.mode_weights) ** 2)

    def _curriculum_steps(self, epoch, max_epochs):
        frac = float(epoch + 1) / float(max_epochs)
        target = int(8 + frac * (self.max_pushforward_steps - 8))
        return max(8, min(self.max_pushforward_steps, target))

    def _teacher_forcing_prob(self, epoch, max_epochs):
        frac = float(epoch) / float(max_epochs - 1)
        return 0.98 - 0.58 * frac

    def train_model(self):
        stop_iter = 0
        patience = 15
        best_valid_rollout_rmse = float("inf")
        num_batches = 20
        train_batch_size = max(1, int(self.ntrain / num_batches))
        valid_batch_size = max(1, int(self.nvalid / num_batches))

        max_epochs = 50
        for i in range(max_epochs):
            rollout_steps = self._curriculum_steps(i, max_epochs)
            tf_prob = self._teacher_forcing_prob(i, max_epochs)
            print(f"[{self.model_tag}] Training iteration:", i)
            print(f"[{self.model_tag}] Rollout steps={rollout_steps}, teacher_forcing_prob={tf_prob:.3f}")
            self.model.train()

            train_loss_accum = 0.0
            train_batches = 0
            for batch in range(num_batches):
                s = batch * train_batch_size
                e = min((batch + 1) * train_batch_size, self.ntrain)
                if s >= e:
                    continue
                xb = self._to_tensor(self.input_seq_train[s:e])
                yb = self._to_tensor(self.output_seq_train[s:e])
                y_roll = self._to_tensor(self.rollout_targets_train[s:e, :rollout_steps, :])

                self.optimizer.zero_grad()
                pred_roll = self.model.rollout(xb, steps=rollout_steps, tf_truth=y_roll, tf_prob=tf_prob)
                one_step_loss = self._weighted_mse(pred_roll[:, 0, :], yb)
                rollout_loss = self._weighted_mse(pred_roll, y_roll)
                loss = 0.2 * one_step_loss + 0.8 * rollout_loss
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
                for batch in range(num_batches):
                    s = batch * valid_batch_size
                    e = min((batch + 1) * valid_batch_size, self.nvalid)
                    if s >= e:
                        continue
                    xb = self._to_tensor(self.input_seq_valid[s:e])
                    yb = self._to_tensor(self.output_seq_valid[s:e])
                    y_roll = self._to_tensor(self.rollout_targets_valid[s:e, :rollout_steps, :])
                    pred_roll = self.model.rollout(xb, steps=rollout_steps, tf_truth=None, tf_prob=0.0)
                    one_step_loss = self._weighted_mse(pred_roll[:, 0, :], yb)
                    rollout_loss = self._weighted_mse(pred_roll, y_roll)
                    valid_loss_accum += float((0.2 * one_step_loss + 0.8 * rollout_loss).item())
                    valid_batches += 1

                valid_loss = valid_loss_accum / max(1, valid_batches)
                full_valid_pred = self.model.rollout(
                    self._to_tensor(self.input_seq_valid), steps=1, tf_truth=None, tf_prob=0.0
                )[:, 0, :].cpu().numpy()

            valid_r2 = coeff_determination(full_valid_pred, self.output_seq_valid)
            valid_rollout_rmse = self._evaluate_rollout_rmse(
                self.input_seq_valid, self.rollout_targets_valid, rollout_steps=self.max_pushforward_steps
            )
            self.train_loss_hist.append(train_epoch_loss)
            self.valid_loss_hist.append(valid_loss)
            self.valid_rollout_rmse_hist.append(valid_rollout_rmse)
            self.scheduler.step(valid_loss)

            if valid_rollout_rmse < best_valid_rollout_rmse:
                print(
                    f"[{self.model_tag}] Improved validation rollout RMSE from:",
                    best_valid_rollout_rmse,
                    " to:",
                    valid_rollout_rmse,
                )
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                best_valid_rollout_rmse = valid_rollout_rmse
                torch.save(self.model.state_dict(), self.ckpt_path)
                stop_iter = 0
            else:
                print(f"[{self.model_tag}] Validation rollout RMSE (no improvement):", valid_rollout_rmse)
                print(f"[{self.model_tag}] Validation R2:", valid_r2)
                stop_iter += 1

            if stop_iter == patience:
                break

        self.model.eval()
        with torch.no_grad():
            test_pred = self.model.rollout(
                self._to_tensor(self.input_seq_test), steps=1, tf_truth=None, tf_prob=0.0
            )[:, 0, :].cpu().numpy()
        test_loss = np.mean(np.square(test_pred - self.output_seq_test))
        test_r2 = coeff_determination(test_pred, self.output_seq_test)
        test_rollout_rmse = self._evaluate_rollout_rmse(
            self.input_seq_test, self.rollout_targets_test, rollout_steps=self.max_pushforward_steps
        )
        print(f"[{self.model_tag}] Test loss:", test_loss)
        print(f"[{self.model_tag}] Test R2:", test_r2)
        print(f"[{self.model_tag}] Test rollout RMSE:", test_rollout_rmse)

        self._plot_training_history()
        self._plot_schematic()

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
                pred_abs = self.model.rollout(
                    self._to_tensor(rec_input_seq), steps=1, tf_truth=None, tf_prob=0.0
                )[0, 0, :].cpu().numpy()
            rec_pred[t] = pred_abs
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
        plt.ylabel("Loss")
        plt.title("Koopman-Style State Space Training History")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Training_Loss_{self.model_tag}.png")
        plt.close()

    def _plot_schematic(self):
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")
        boxes = [
            (0.02, 0.35, 0.16, 0.3, f"Input State\\n(r={self.state_len})"),
            (0.22, 0.35, 0.16, 0.3, "Encoder\\nMLP"),
            (0.42, 0.35, 0.16, 0.3, "Latent Dynamics\\n z' = Kz + g(z)"),
            (0.62, 0.35, 0.16, 0.3, "Decoder\\nMLP"),
            (0.82, 0.35, 0.16, 0.3, "Next State\\nRollout"),
        ]
        for x, y, w, h, txt in boxes:
            rect = plt.Rectangle((x, y), w, h, facecolor="#dce7f7", edgecolor="#295c9a", linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
        arrows = [(0.18, 0.5, 0.22, 0.5), (0.38, 0.5, 0.42, 0.5), (0.58, 0.5, 0.62, 0.5), (0.78, 0.5, 0.82, 0.5)]
        for x1, y1, x2, y2 in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#1f2a24"))
        ax.text(
            0.5,
            0.08,
            "Koopman-style neural state space with stable latent linear operator and nonlinear residual",
            ha="center",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(f"Torch_LSTM_Schematic_{self.model_tag}.png")
        plt.close()

    def _evaluate_rollout_rmse(self, input_seq, rollout_targets, rollout_steps):
        self.model.eval()
        rollout_steps = min(rollout_steps, rollout_targets.shape[1])
        sqerr = []
        with torch.no_grad():
            for i in range(input_seq.shape[0]):
                seed = self._to_tensor(input_seq[i : i + 1])
                pred = self.model.rollout(seed, steps=rollout_steps, tf_truth=None, tf_prob=0.0)[0].cpu().numpy()
                truth = rollout_targets[i, :rollout_steps, :]
                sqerr.append(np.mean(np.square(pred - truth)))
        return float(np.sqrt(np.mean(sqerr)))

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
    def __init__(self, state_len, hidden_size=50):
        super().__init__()
        self.l1 = nn.LSTM(input_size=state_len, hidden_size=hidden_size, batch_first=True)
        self.l2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, state_len)

    def forward(self, x):
        h1, _ = self.l1(x)
        h2, _ = self.l2(h1)
        return self.out(h2[:, -1, :])


class standard_lstm:
    def __init__(self, data):
        np.random.seed(7)
        torch.manual_seed(7)

        self.device = torch.device("cpu")
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]

        self.preproc_pipeline = Pipeline(
            [
                ("stdscaler", StandardScaler()),
                ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ]
        )
        self.data = self.preproc_pipeline.fit_transform(data)

        self.seq_num = 5
        self.total_size = np.shape(data)[0] - int(self.seq_num)

        input_seq = np.zeros((self.total_size, self.seq_num, self.state_len), dtype=np.float32)
        output_seq = np.zeros((self.total_size, self.state_len), dtype=np.float32)

        for t in range(self.total_size):
            input_seq[t, :, :] = self.data[t : t + self.seq_num, :]
            output_seq[t, :] = self.data[t + self.seq_num, :]

        idx = np.arange(self.total_size)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]

        split_test = int(0.9 * self.total_size)
        self.input_seq_test = input_seq[split_test:]
        self.output_seq_test = output_seq[split_test:]
        input_seq = input_seq[:split_test]
        output_seq = output_seq[:split_test]

        self.ntrain = int(0.8 * np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[: self.ntrain]
        self.output_seq_train = output_seq[: self.ntrain]
        self.input_seq_valid = input_seq[self.ntrain :]
        self.output_seq_valid = output_seq[self.ntrain :]

        self.model = TorchLSTMNet(state_len=self.state_len).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train_loss_hist = []
        self.valid_loss_hist = []

        self.ckpt_path = "./checkpoints/my_checkpoint.pt"
        os.makedirs("./checkpoints", exist_ok=True)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def train_model(self):
        stop_iter = 0
        patience = 10
        best_valid_loss = float("inf")

        self.num_batches = 20
        self.train_batch_size = max(1, int(self.ntrain / self.num_batches))
        self.valid_batch_size = max(1, int(self.nvalid / self.num_batches))

        for i in range(10):
            print("Training iteration:", i)
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

                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
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
                    pred = self.model(xb)
                    valid_loss_accum += float(self.criterion(pred, yb).item())
                    valid_batches += 1

                valid_loss = valid_loss_accum / max(1, valid_batches)
                full_valid_pred = self.model(self._to_tensor(self.input_seq_valid)).cpu().numpy()

            valid_r2 = coeff_determination(full_valid_pred, self.output_seq_valid)
            self.train_loss_hist.append(train_epoch_loss)
            self.valid_loss_hist.append(valid_loss)

            if valid_loss < best_valid_loss:
                print("Improved validation loss from:", best_valid_loss, " to:", valid_loss)
                print("Validation R2:", valid_r2)
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.ckpt_path)
                stop_iter = 0
            else:
                print("Validation loss (no improvement):", valid_loss)
                print("Validation R2:", valid_r2)
                stop_iter += 1

            if stop_iter == patience:
                break

        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(self._to_tensor(self.input_seq_test)).cpu().numpy()
        test_loss = np.mean(np.square(test_pred - self.output_seq_test))
        test_r2 = coeff_determination(test_pred, self.output_seq_test)
        print("Test loss:", test_loss)
        print("Test R2:", test_r2)

        self._plot_training_history()
        self._plot_torch_lstm_schematic()

    def restore_model(self):
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval()
        print("Model restored successfully!")

    def model_inference(self, test_data):
        self.restore_model()

        test_data = self.preproc_pipeline.transform(test_data)
        test_total_size = np.shape(test_data)[0] - int(self.seq_num)

        rec_input_seq = test_data[: self.seq_num, :].reshape(1, self.seq_num, self.state_len).astype(np.float32)

        rec_output_seq = np.zeros((test_total_size, self.state_len), dtype=np.float32)
        for t in range(test_total_size):
            rec_output_seq[t, :] = test_data[t + self.seq_num, :]

        print("Making predictions on testing data")
        rec_pred = np.copy(rec_output_seq)
        for t in range(test_total_size):
            with torch.no_grad():
                pred = self.model(self._to_tensor(rec_input_seq)).cpu().numpy()[0]
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
            plt.savefig("Mode_" + str(i) + "_prediction.png")
            plt.close()

        return rec_output_seq, rec_pred

    def _plot_training_history(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.train_loss_hist, marker="o", label="Train Loss")
        plt.plot(self.valid_loss_hist, marker="s", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Torch LSTM Training History")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Training_Loss.png")
        plt.close()

    def _plot_torch_lstm_schematic(self):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")

        boxes = [
            (0.03, 0.35, 0.18, 0.3, "Input Sequence\n(L=5, r=3)"),
            (0.27, 0.35, 0.18, 0.3, "LSTM Layer 1\n(hidden=50)"),
            (0.51, 0.35, 0.18, 0.3, "LSTM Layer 2\n(hidden=50)"),
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

        ax.text(0.5, 0.08, "Autoregressive rollout: predicted mode coefficients are fed back as next-step inputs", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig("Torch_LSTM_Schematic.png")
        plt.close()


if __name__ == "__main__":
    print("Torch architecture file")

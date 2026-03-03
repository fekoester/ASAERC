from __future__ import annotations


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class PDETrainer:
    def __init__(
        self, readout, pde_reservoir, lr: float = 1e-3, final_lr: float = 1e-4, device=None
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.readout = readout.to(self.device)
        self.pde_reservoir = pde_reservoir
        self.initial_lr = lr
        self.final_lr = final_lr

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.readout.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1, eta_min=final_lr)

    def train_loop(self, dataloader, n_epochs: int = 10, repulsion_coef: float = 0.0):
        """
        Trains readout on tuples (meas_fixed(u_{t+T}), field(u_{t+T}), target).
        This guarantees adaptive sensing samples the correct u_{t+T} for each sample,
        even when the dataloader shuffles.
        """
        self.readout.train()
        data_loss_history = []
        repel_loss_history = []

        self.scheduler.T_max = n_epochs

        for epoch in range(n_epochs):
            total_data, total_repel, total_count = 0.0, 0.0, 0

            for batch_meas, batch_field, batch_targets in dataloader:
                batch_meas = batch_meas.to(self.device)
                batch_field = batch_field.to(self.device)
                batch_targets = batch_targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if repulsion_coef > 0.0:
                    preds, query_xy, _ = self.readout(batch_meas, batch_field, return_queries=True)
                else:
                    preds = self.readout(batch_meas, batch_field)

                data_loss = self.criterion(preds, batch_targets)
                repel_loss = torch.tensor(0.0, device=self.device)

                if repulsion_coef > 0.0 and query_xy is not None:
                    # query repulsion (encourage spread)
                    B, Q, _ = query_xy.shape
                    diff = query_xy.unsqueeze(2) - query_xy.unsqueeze(1)
                    dist2 = (diff.pow(2).sum(-1)).clamp(min=1e-6)
                    inv = 1.0 / dist2
                    idx = torch.triu_indices(Q, Q, offset=1, device=self.device)
                    repel_loss = inv[:, idx[0], idx[1]].mean()
                    loss = data_loss + repulsion_coef * repel_loss
                else:
                    loss = data_loss

                loss.backward()
                self.optimizer.step()

                bs = batch_meas.size(0)  # FIX: was batch_states (undefined)
                total_data += float(data_loss.item()) * bs
                total_repel += float(repel_loss.item()) * bs
                total_count += bs

            self.scheduler.step()

            avg_data = total_data / max(1, total_count)
            avg_repel = total_repel / max(1, total_count)
            data_loss_history.append(avg_data)
            repel_loss_history.append(avg_repel)

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[epoch {epoch + 1:4d}/{n_epochs}] loss={avg_data:.6f} repel={avg_repel:.6f} lr={lr:.3e}"
            )

        return data_loss_history, repel_loss_history

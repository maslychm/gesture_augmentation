import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from sys import platform
BARS_ENABLED = False if platform == "linux" else True

class LitGestureNN(pl.LightningModule):
    def __init__(self, batch_size, num_classes, learning_rate):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.gru = nn.GRU(2, 96, 2, batch_first=True, dropout=0.3)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(96),
            nn.Dropout(0.3),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.25),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x_padded, x_lengths):
        x_packed = pack_padded_sequence(x_padded, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(x_packed)
        y_hat = self.classifier(hidden[-1])
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        y_hat = self(x, x_lengths)
        loss = F.cross_entropy(y_hat, y)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=BARS_ENABLED)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        y_hat = self(x, x_lengths)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat.argmax(dim=1), y, task="multiclass", num_classes=self.num_classes)
        # keeping on_epoch as True because EarlyStopping needs to know when to stop
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=BARS_ENABLED)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=BARS_ENABLED)
        return {"val_perf": {"val/loss": loss, "val/acc": acc}}
    
    def test_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        y_hat = self(x, x_lengths)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat.argmax(dim=1), y, task="multiclass", num_classes=self.num_classes)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=BARS_ENABLED)
        return {"test_perf": {"test/loss": loss, "test/acc": acc}}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
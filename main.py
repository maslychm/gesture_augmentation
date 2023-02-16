import math
from typing import List
from model import LitGestureNN
from data import load_data
from options import Options

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from sys import platform

"""
Example calls:

python main.py
python main.py --augm None
python main.py --augm gaussian_frame-skip_rotate --sub_idx 2 --original_per_class 2 --synthetic_per_class 100
python main.py --condition ui --augm avc --num_participants 2 --original_per_class 2
"""

def test_model(trainer: Trainer, testing_dataloader, callbacks: List[Callback]):

    for callback in callbacks:

        if not isinstance(callback, ModelCheckpoint):
            continue

        print(f"\n TESTING WITH {callback.monitor} CHECKPOINT \n")
        callback_results = trainer.test(
            dataloaders=testing_dataloader,
            ckpt_path=callback.best_model_path,
        )
        print(callback_results)


def train_model(opt: Options, train_dataloader, val_dataloader):

    model = LitGestureNN(opt.batch_size, opt.num_classes, opt.learning_rate)
    patience_epochs = 150
    if opt.batch_size < len(train_dataloader.dataset):
        num_train_batches = math.ceil(
            len(train_dataloader.dataset) / opt.batch_size)
        patience_epochs = int(patience_epochs / num_train_batches)

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min"),
        EarlyStopping(monitor="val/loss", min_delta=0.001,
                      patience=patience_epochs, mode="min"),
    ]

    trainer = Trainer(
        max_steps=2000,
        accelerator=None if platform == "darwin" else "gpu",
        devices=None if platform == "darwin" else 1,
        deterministic=opt.fixed,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        logger=False
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return model, trainer, callbacks

def main():
    opt = Options().parse()

    if opt.fixed:
        seed_everything(42)

    train_dataloader, val_dataloader, test_dataloader = load_data(opt)
    model, trainer, callbacks = train_model(opt, train_dataloader, val_dataloader)
    test_model(trainer, test_dataloader, callbacks)

if __name__ == "__main__":
    main()

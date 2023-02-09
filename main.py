import math
from typing import List
from model import LitGestureNN

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from dataset import GDSDataset

from synthesis import DataFactory
from utils import first_point_to_origin_whole_set
from torch_utils import encode_labels, get_dataloaders

from sys import platform

# params
BATCH_SIZE = 512
NUM_CLASSES = 16
LEARNING_RATE = 0.001
SYNTHETIC_PER_CLASS = 1000
SYNTHETIC_VALIDATION = False
FIXED = True


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


def load_data(sub_idx=1, num_classes=16, synthetic_per_class=100, synthetic_validation=False, batch_size=512):
    dataset = GDSDataset("./", sub_idx=sub_idx)
    train, val, test = dataset.ud_split(k=1, fixed=True)
    print(
        f"UD - Original train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    synthetic_per_sample = synthetic_per_class // (len(train) // num_classes)
    synth = DataFactory.generate_avc(train, n=synthetic_per_sample)
    train.extend(synth)

    if synthetic_validation:
        test.extend(val)
        val = DataFactory.generate_avc(train, n=synthetic_per_sample*2)

    train, val, test = first_point_to_origin_whole_set(train, val, test)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = encode_labels(
        train, val, test
    )

    return get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)


def train_model(train_dataloader, val_dataloader):

    if FIXED:
        seed_everything(42)

    model = LitGestureNN(BATCH_SIZE, NUM_CLASSES, LEARNING_RATE)
    patience_epochs = 150
    if BATCH_SIZE < len(train_dataloader.dataset):
        num_train_batches = math.ceil(
            len(train_dataloader.dataset) / BATCH_SIZE)
        patience_epochs = int(patience_epochs / num_train_batches)

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min"),
        EarlyStopping(monitor="val/loss", min_delta=0.001,
                      patience=patience_epochs, mode="min"),
    ]

    trainer = Trainer(
        max_steps=2000,
        gpus=None if platform == "darwin" else 1,
        deterministic=FIXED,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return model, trainer, callbacks


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = load_data(
        synthetic_per_class=300)
    model, trainer, callbacks = train_model(train_dataloader, val_dataloader)
    test_model(trainer, test_dataloader, callbacks)

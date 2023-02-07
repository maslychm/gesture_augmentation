import torch
from torch import tensor

from utils import LabelEncoder

class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.lengths = [len(x) for x in X]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


class PadCollate:
    """
    A DataLoader collation function that zero-pads to the length of the longest sequence in a batch
    """

    def __call__(self, batch):
        # Sort based on the length of each sequence
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        xs = [x[0] for x in sorted_batch]
        ys = [x[1] for x in sorted_batch]
        xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)
        lengths = torch.LongTensor([len(x) for x in xs])
        return xs_padded, lengths, ys


def encode_labels(train, val, test):
    train_x = [x[0] for x in train]
    train_y = [x[1] for x in train]
    val_x = [x[0] for x in val]
    val_y = [x[1] for x in val]
    test_x = [x[0] for x in test]
    test_y = [x[1] for x in test]

    le = LabelEncoder()
    le.fit(train_y)

    y_train = le.transform(train_y)
    y_val = le.transform(val_y)
    y_test = le.transform(test_y)

    X_train = [tensor(x, dtype=torch.float32) for x in train_x]
    X_val = [tensor(x, dtype=torch.float32) for x in val_x]
    X_test = [tensor(x, dtype=torch.float32) for x in test_x]

    y_train = tensor(y_train, dtype=torch.long)
    y_val = tensor(y_val, dtype=torch.long)
    y_test = tensor(y_test, dtype=torch.long)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
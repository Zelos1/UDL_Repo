import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd


def load_mnist(val_size=100, init_size=20):
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    x_train = train_dataset.data.float() / 255.0 
    x_test  = test_dataset.data.float() / 255.0
    x_train = x_train.unsqueeze(1)
    x_test  = x_test.unsqueeze(1)

    y_train = train_dataset.targets
    y_test  = test_dataset.targets

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples, before reduction")
    print(x_test.shape[0], "test samples")

    num_classes = 10

    y_train_1h = F.one_hot(y_train, num_classes=num_classes)
    y_test_1h  = F.one_hot(y_test,  num_classes=num_classes)

    x_train_indices = np.zeros(init_size, dtype=int)
    x_val_indices   = np.zeros(val_size, dtype=int)

    for i in range(num_classes):
        idx = torch.where(y_train == i)[0].numpy()
        selected = np.random.choice(idx, size=12, replace=False)

        x_train_indices[i*2:i*2+2] = selected[:2]
        x_val_indices[i*10:i*10+10] = selected[2:]

    # Extract new subsets
    x_train_new = x_train[x_train_indices]
    y_train_new = y_train_1h[x_train_indices]

    x_val = x_train[x_val_indices]
    y_val = y_train_1h[x_val_indices]

    # Pool remaining data (remove selected indices)
    remaining = np.setdiff1d(np.arange(len(x_train)),
                            np.concatenate((x_train_indices, x_val_indices)))

    X_p = x_train[remaining]
    y_p = y_train_1h[remaining]

    return x_train_new, y_train_new, X_p, y_p, x_val, y_val, x_test, y_test_1h

def load_uci(val_size=100, init_size=20):
    air_quality = fetch_ucirepo(id=360)
    target_cols = ["CO(GT)", "NO2(GT)", "PT08.S5(O3)"]
    df = air_quality.data.features
    df.replace(-200, np.nan, inplace=True)
    df = df.dropna(subset=target_cols)
  
    # data (as pandas dataframes) 
    y = df[target_cols].to_numpy()
    X = df.drop(columns=target_cols + ["Date", "Time"], errors="ignore").to_numpy()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train = y_train
    y_test  = y_test

    x_train_indices = np.zeros(init_size, dtype=int)
    x_val_indices   = np.zeros(val_size, dtype=int)

    selected = np.random.choice(range(len(y_train)), size=val_size + init_size, replace=False)

    x_train_indices = selected[:init_size]
    x_val_indices = selected[init_size:]

    x_train_new = x_train[x_train_indices]
    y_train_new = y_train[x_train_indices]

    x_val = x_train[x_val_indices]
    y_val = y_train[x_val_indices]

    remaining = np.setdiff1d(
        np.arange(len(x_train)),
        np.concatenate((x_train_indices, x_val_indices))
    )

    X_p = x_train[remaining]
    y_p = y_train[remaining]
    return x_train_new, y_train_new, X_p, y_p, x_val, y_val, x_test, y_test

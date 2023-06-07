import pandas as pd
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import DRDataset
from utilities.utilities import df_train, df_validation, PathConstant


def load_datasets(train_config):
    train, validation = df_train(), df_validation()

    train_ds = DRDataset(
        images_path=PathConstant.train_path,
        df=train,
        transform=train_config.transform
    )

    val_ds = DRDataset(
        images_path=PathConstant.train_path,
        df=validation,
        transform=train_config.validation_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        shuffle=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        shuffle=False
    )
    return train_loader, val_loader


def create_predictions(loader, model, loss_fn, device="cuda"):
    model.eval()
    all_predictions, all_values, all_names, all_losses = [], [], [], []

    for batch, value, image_name in tqdm(loader):
        batch = batch.to(device=device)
        value = value.to(device=device).view(-1)

        with torch.no_grad():
            predictions = model(batch)
        loss = loss_fn(predictions, value)
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.long().view(-1)

        all_predictions.append(predictions.detach().cpu().numpy())
        all_values.append(value.detach().cpu().numpy())
        all_names.append(image_name)
        all_losses.append(loss.cpu().numpy())

    model.train()

    return np.concatenate(all_predictions, axis=0, dtype=np.int64), \
           np.concatenate(all_values, axis=0, dtype=np.int64), \
           np.concatenate(all_names, axis = 0), \
           np.mean(all_losses)


def evaluate_model(epoch, val_loader, model, config, loss_fn):
    predictions, labels, _, loss = create_predictions(val_loader, model, loss_fn, device = config.device)
    kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
    accuracy = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average="macro")

    print(f"Cohen Kappa Score: {kappa}")
    if config.writer:
        config.writer.add_scalar("Kappa", kappa, epoch)
        config.writer.add_scalar("Accuracy", accuracy, epoch)
        config.writer.add_scalar("F1", f1, epoch)
        config.writer.add_scalar("val_loss", loss, epoch)
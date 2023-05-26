import numpy as np
import torch
from tqdm import tqdm

from efficientnet.config import model_config, train_config

from model.config import TrainConfig, ModelConfig
from utils import (
    load_checkpoint,
    save_checkpoint,
)

def train_one_epoch(epoch, loader, model, optimizer, loss_fn, scaler, device,
                    writer = None, scheduler = None):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, names) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    if scheduler is not None:
        scheduler.step()

    average_loss = np.mean(losses)
    print(f"Loss average over epoch: {average_loss}")
    if writer:
        writer.add_scalar('loss', average_loss, global_step = epoch)



def main(model_config : ModelConfig, train_config : TrainConfig):
    train_loader, val_loader = train_config.load_database(train_config)

    model = model_config.model.to(train_config.device)
    loss_fn = model_config.loss_fn.to(train_config.device)
    optimizer = model_config.optimizer
    scaler = model_config.scaler
    scheduler = model_config.scheduler

    if train_config.load_model:
        print(f"Loading model from {train_config.checkpoint_file}")
        load_checkpoint(torch.load(train_config.checkpoint_file), model, optimizer,
                        train_config.learning_parameters['lr'])

    writer = train_config.writer
    for epoch in range(train_config.num_epochs):
        train_one_epoch(epoch, train_loader, model, optimizer, loss_fn, scaler, train_config.device,
                        writer = writer, scheduler = scheduler)
        train_config.evaluate_model(epoch, val_loader, model, train_config, loss_fn)

        if train_config.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            save_checkpoint(checkpoint, filename=train_config.save_model_prefix + f"_{epoch}.pth.tar")

if __name__ == "__main__":
    main(model_config, train_config)

import torch
from torch import nn


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict = False)

    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def probability_from_logits(logits):
    probabilities = nn.functional.softmax(logits)

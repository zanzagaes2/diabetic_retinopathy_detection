from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ModelConfig:
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : torch.nn.Module
    scaler : torch.cuda.amp.GradScaler = None
    scheduler : torch.optim.lr_scheduler = None

@dataclass
class TrainConfig:
    device : str

    learning_parameters : dict
    batch_size : int
    num_epochs : int
    num_workers : int
    pin_memory : bool
    
    load_model : bool
    checkpoint_file : str

    save_model : bool
    save_model_prefix : str

    load_database : Callable

    writer : Callable

    evaluate_model : Callable = None
    transform : Callable = None
    validation_transform : Callable = None

@dataclass
class TestConfig:
    device : str

    batch_size : int
    num_workers : int
    pin_memory : bool

    checkpoint_file : str
    transform : Callable

    create_predictions : Callable
    output_file : str

    load_database : Callable

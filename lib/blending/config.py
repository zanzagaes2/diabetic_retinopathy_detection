import timm
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from efficientnet.utils import test_loader
from model.config import TrainConfig, ModelConfig, TestConfig
from blending.utils import evaluate_model, create_predictions, load_datasets


def create_model_config():
        model = nn.Sequential(
                nn.Linear(50, 500),
                nn.LeakyReLU(0.1),
                nn.Linear(500, 500),
                nn.Dropout(p=0.6),
                nn.LeakyReLU(0.1),
                # nn.Linear(500, 500),
                # nn.Dropout(p=0.6),
                # nn.LeakyReLU(0.5),
                nn.Linear(500, 2)
        )
        optimizer = optim.AdamW(
                model.parameters(),
                **train_config.learning_parameters
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                 step_size = 40, gamma = 0.8
        )

        return ModelConfig(
                model = model,
                loss_fn = nn.CrossEntropyLoss(),
                optimizer = optimizer,
                scheduler = scheduler,
                scaler = torch.cuda.amp.GradScaler()
        )

train_config = TrainConfig(
        device = "cuda",

        learning_parameters = {
                'lr' : 0.00002,
                'weight_decay' : 0.05
        },

        batch_size = 32,
        num_epochs = 300,
        num_workers = 4,
        pin_memory = True,

        load_model=False,
        checkpoint_file = "./checkpoints/blending_model/exper__8.pth.tar",

        save_model = True,
        save_model_prefix = "./checkpoints/blending_model/exper_",

        load_database = load_datasets,
        load_database_info = {
                'train_dataset': './predictions/pre_blending/predictions_validation.csv',
                'val_dataset': './predictions/pre_blending/predictions_test.csv'
        },

        writer = SummaryWriter('../logs/fit/blending_model_exper'),
        evaluate_model = evaluate_model
)

test_config = TestConfig(
        device = "cuda",

        batch_size = 32,
        num_workers = 4,
        pin_memory = True,

        checkpoint_file = "",
        output_file="",

        create_predictions=create_predictions,
        transform=None,

        load_database = test_loader,
)

model_config = create_model_config()


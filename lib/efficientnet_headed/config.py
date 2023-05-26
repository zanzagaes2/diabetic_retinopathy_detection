import timm
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from efficientnet.config import rotate
from efficientnet.transformations import train_transform
from model.config import TrainConfig, ModelConfig, TestConfig
from efficientnet.utils import evaluate_model, load_datasets, \
        create_predictions_probability, test_loader


def create_model_config():
        model = timm.create_model('tf_efficientnetv2_b3', pretrained = True, num_classes = 0)

        for param in model.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
                nn.Linear(1536, 384),
                nn.LeakyReLU(0.5),
                nn.Dropout(p=0.3),
                nn.Linear(384, 96),
                nn.LeakyReLU(0.5),
                nn.Dropout(p=0.3),
                nn.Linear(96, 32),
                nn.LeakyReLU(0.5),
                nn.Linear(32, 5)
        )

        optimizer = optim.Adamax(
                model.parameters(),
                **train_config.learning_parameters
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                 T_max = 50
        )

        return ModelConfig(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                loss_fn = nn.CrossEntropyLoss(),
                scaler = torch.cuda.amp.GradScaler()
        )

train_config = TrainConfig(
        device = "cuda",

        learning_parameters = {
                'lr': 0.002
        },

        batch_size = 64,
        num_epochs = 90,
        num_workers = 4,
        pin_memory = True,

        load_model=True,
        save_model = True,

        load_database = load_datasets,

        evaluate_model = evaluate_model,
        transform= train_transform,
        validation_transform = train_transform,
)

test_config = TestConfig(
        device = "cuda",

        batch_size = 128,
        num_workers = 4,
        pin_memory = True,

        checkpoint_file = "./checkpoints/efficientnetv2b3_head/leaky_dropout_41.pth.tar",
        output_file="./predictions/pre_blending/predictions_validation.csv",

        create_predictions=create_predictions_probability,
        transform=rotate,

        load_database = test_loader,
)

model_config = create_model_config()


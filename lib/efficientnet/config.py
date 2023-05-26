import timm
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from efficientnet.transformations import rotate, train_transform
from utilities.transformations import NormalizingConstants, normalize, light_augmentation_and_normalize, \
        rotate_and_normalize
from model.config import TrainConfig, ModelConfig, TestConfig
from efficientnet.utils import evaluate_model, create_predictions, load_datasets


def create_model_config():
        # model = timm.create_model('tf_efficientnetv2_b3', pretrained = True, num_classes = 5)
        model = timm.create_model('convnext_tiny', pretrained = True, num_classes = 5)
        return ModelConfig(
                model = model,
                optimizer = optim.Adamax(
                        model.parameters(),
                        **train_config.learning_parameters
                ),
                loss_fn = nn.CrossEntropyLoss(),
                scaler = torch.cuda.amp.GradScaler()
        )

train_config = TrainConfig(
        device = "cuda",

        learning_parameters = {
                # 'lr': 0.00001,
                'lr': 0.0001,
                'weight_decay' : 0.005
        },

        batch_size = 10,
        num_epochs = 90,
        num_workers = 4,
        pin_memory = True,

        load_model=False,
        save_model = True,

        load_database = load_datasets,

        evaluate_model = evaluate_model,
        transform= train_transform,
        validation_transform = train_transform,
)

test_config = TestConfig(
        device = "cuda",

        batch_size = 32,
        num_workers = 4,
        pin_memory = True,

        create_predictions=create_predictions,
        transform=rotate,

        load_database = load_datasets,
)

model_config = create_model_config()


import timm
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from efficientnet.transformations import train_transform, normalize_image, rotate
from efficientnet.utils import test_loader, create_predictions_probability
from model.config import TrainConfig, ModelConfig, TestConfig
from blending.utils import evaluate_model, load_datasets
from efficientnet.config import train_config as base_config

from model.utils import load_checkpoint


def create_model_config():

        class Blender(torch.nn.Module):
                def __init__(self, base, classifier):
                        super().__init__()
                        self.base = base
                        self.classifier = classifier

                def forward(self, x):
                        N = len(x)
                        out = self.base(x)
                        w = torch.cat((out[::2], out[1::2]), dim = 1)
                        logits = self.classifier(w)
                        logits = torch.reshape(logits, (N, 5))
                        return logits

        base = timm.create_model('tf_efficientnetv2_b3',
                                 pretrained = True, num_classes = 0,
                                 drop_rate=0.2)

        print(f"=> Loading checkpoint: {base_config.checkpoint_file}")
        load_checkpoint(torch.load(base_config.checkpoint_file), base, None, None)
        for param in base.parameters():
                param.requires_grad = True

        classifier = nn.Sequential(
                nn.Linear(3072, 1536),
                nn.LeakyReLU(0.5),
                nn.Dropout(p=0.3),
                nn.Linear(1536, 384),
                nn.LeakyReLU(0.5),
                nn.Dropout(p=0.3),
                nn.Linear(384, 96),
                nn.LeakyReLU(0.5),
                nn.Dropout(p=0.3),
                nn.Linear(96, 32),
                nn.LeakyReLU(0.5),
                nn.Linear(32, 10)
        )

        model = Blender(base, classifier)

        optimizer = optim.AdamW(
                model.parameters(),
                **train_config.learning_parameters
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                 step_size = 10, gamma = 0.5
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
                'lr' : 5e-5,
                'weight_decay' : 0.001
        },

        batch_size = 16,
        num_epochs = 300,
        num_workers = 4,
        pin_memory = True,

        load_model=False,
        checkpoint_file = "./checkpoints/blending_model/exper_classifier_50.pth.tar",

        save_model = True,
        save_model_prefix = "./checkpoints/blending_model/exper_unfrozen",

        load_database = load_datasets,

        writer = SummaryWriter('../logs/fit/blending_model_exper_unfrozen'),
        evaluate_model = evaluate_model,
        transform= train_transform,
        validation_transform = normalize_image,
)

test_config = TestConfig(
        device = "cuda",

        batch_size = 128,
        num_workers = 4,
        pin_memory = True,

        checkpoint_file = "",
        output_file="",

        create_predictions=create_predictions_probability,
        transform= rotate,

        load_database = test_loader,
)

model_config = create_model_config()

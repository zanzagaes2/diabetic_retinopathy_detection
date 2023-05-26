import timm
import torch

from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from MIL.models.MIL_VT import *
from efficientnet.transformations import train_transform, normalize_image
from efficientnet.utils import test_loader, create_predictions, evaluate_model, load_datasets
from model.config import TrainConfig, ModelConfig, TestConfig

def create_model_config():
        base_model = "MIL_VT_small_patch16_512"
        # model = MIL_VT_FineTune(base_model)
        model = timm.create_model(model_name=base_model,
            pretrained=False,
            num_classes=5,
            drop_rate=0.2,
            drop_path_rate=0.1,
            drop_block_rate=None,
        )

        unfrozen_modules = [
                'head',
                'MIL_Prep',
                'MIL_attention',
                'MIL_classifier'
        ]
        for param in model.parameters():
                param.requires_grad = False
        
        for block in model.blocks:
                for param in block.attn.parameters():
                      param.requires_grad = True
        for module in unfrozen_modules:
                for param in model.get_submodule(module).parameters():
                        param.requires_grad = True

        optimizer = optim.AdamW(
                model.parameters(),
                **train_config.learning_parameters
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                 step_size = 10, gamma = 0.5
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
                "lr" : 2e-6,
                "weight_decay": 2e-4
        },

        batch_size = 10,
        num_epochs = 150,
        num_workers = 4,
        pin_memory = True,

        load_model= True,
        checkpoint_file = "./checkpoints/transformer/unfrozen_all_14.pth.tar",

        save_model = True,
        save_model_prefix = "./checkpoints/transformer/unfrozen_all",

        load_database = load_datasets,

        writer = SummaryWriter('../logs/fit/transformer_unfrozen/'),

        evaluate_model = evaluate_model,
        transform= train_transform,
        validation_transform = normalize_image,
)

test_config = TestConfig(
        device = "cuda",

        batch_size = 128,
        num_workers = 4,
        pin_memory = True,

        checkpoint_file = "./checkpoints/efficientnetv2b3_classifier/_10.pth.tar",
        output_file="./predictions/pre_blending/efficientnetv2b3_classifier_test.csv",

        create_predictions=create_predictions,
        transform=normalize_image,

        load_database = test_loader,
)

model_config = create_model_config()


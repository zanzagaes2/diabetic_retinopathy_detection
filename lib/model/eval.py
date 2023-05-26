import torch
from sklearn.metrics import cohen_kappa_score

from utils import (
    load_checkpoint,
)
from model.config import ModelConfig, TestConfig
from efficientnet.config import model_config as efficientnet_model_config, \
    test_config as efficientnet_test_config
from pandas import DataFrame as df

def main(model_config : ModelConfig, test_config : TestConfig):
    test = test_config.load_database(test_config)
    model = model_config.model.to(test_config.device)

    for epoch in range(9, 31):
        checkpoint_file = f"./checkpoints/efficientnetv2b3_classifier/_{epoch}.pth.tar"
        model.eval()
        print(f"Loading model from {checkpoint_file}")
        load_checkpoint(torch.load(checkpoint_file), model, None, None)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                predictions, labels, _, loss = test_config.create_predictions(test, model, model_config.loss_fn)
                kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
                print(f"epoch: {epoch} -- kappa: {kappa} -- test loss: {loss}")

if __name__ == "__main__":
    main(efficientnet_model_config, efficientnet_test_config)

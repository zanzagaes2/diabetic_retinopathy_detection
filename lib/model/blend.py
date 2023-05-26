import torch
import numpy as np
from pandas import DataFrame as df

from blending.config import model_config as blending_model_config, \
    test_config as blending_test_config
from model.config import ModelConfig, TestConfig
from utils import (
    load_checkpoint,
)

def main(model_config : ModelConfig, test_config : TestConfig):
    test = test_config.load_database(test_config)
    model = model_config.model.to(test_config.device)

    print(f"Loading model from {test_config.checkpoint_file}")
    load_checkpoint(torch.load(test_config.checkpoint_file), model, None, None)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            predictions, values, names, _ = test_config.create_predictions(test, model, model_config.loss_fn)
    predictions = np.concatenate(predictions)
    values = np.concatenate(values)
    data = df({'prediction': predictions, 'value': values}, index = names)
    data.to_csv(test_config.output_file, sep = ",")

if __name__ == "__main__":
    main(blending_model_config, blending_test_config)

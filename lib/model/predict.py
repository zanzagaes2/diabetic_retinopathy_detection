import torch
import numpy as np

from utilities.utilities import df_test
from utils import (
    load_checkpoint,
)
from model.config import ModelConfig, TestConfig
from blending.experimental import model_config, test_config
from pandas import DataFrame as df

def main(model_config : ModelConfig, test_config : TestConfig):
    test = test_config.load_database(test_config)
    model = model_config.model.to(test_config.device)

    print(f"Loading model from {test_config.checkpoint_file}")
    load_checkpoint(torch.load(test_config.checkpoint_file), model, None, None)

    model.eval()
    times = 5
    predictions = np.zeros([times, 53572, 5])
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(times):
                preds, values, names = \
                    test_config.create_predictions(test, model, model_config.loss_fn)
                predictions[i] = preds
    np.save("./predictions/blended/predictions_test_head", predictions)
    data = df({'value': values}, index = names)
    data.to_csv(test_config.output_file, sep = ",")

if __name__ == "__main__":
    main(model_config, test_config)
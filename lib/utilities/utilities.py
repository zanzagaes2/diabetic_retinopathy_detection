import os
from functools import partial

import pandas as pd

from enum import IntEnum

class PathConstant:
    # Fill with the corresponding path
    train_csv = '../dataset/trainLabels.csv'
    test_csv = '../dataset/testLabels.csv'
    train_path = '../dataset/train'
    test_path = '../dataset/test'

def load_df_and_append_extension(csv_path, extension):
    with open(csv_path) as file:
        df = pd.read_csv(file)
        df.image += extension
        return df


def df_train():
    df = load_df_and_append_extension(PathConstant.train_csv, extension='.jpeg')
    N = round(len(df) * .8)
    N += (N % 2)
    return df[:N]

def df_validation():
    df = load_df_and_append_extension(PathConstant.train_csv, extension='.jpeg')
    N = round(len(df) * .8)
    N += (N % 2)
    return df[N:]

df_test = partial(load_df_and_append_extension, PathConstant.test_csv, extension = '.jpeg')

class Level(IntEnum):
    healthy = 0
    light = 1
    medium = 2
    severe = 3
    very_severe = 4

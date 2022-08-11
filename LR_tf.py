import pandas as pd
import numpy as np
import tensorflow as tf


def create_TFdataset():
    df_train = pd.read_csv("dx_train.csv", encoding="utf-8")
    df_val = pd.read_csv("dx_test.csv", encoding="utf-8")
    print(df_train.head())

create_TFdataset()
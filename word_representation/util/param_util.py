import argparse
import codecs
import json

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams"]

def create_default_hyperparams():
    """create default hyperparameters"""
    hyperparams = tf.contrib.training.HParams()
    return hyperparams

def load_hyperparams(config_file):
    """load hyperparameters from config file"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams = create_default_hyperparams()
            hyperparams_dict = json.load(file)
            hyperparams.set_from_map(hyperparams_dict)
            
            return hyperparams
    else:
        raise FileNotFoundError("config file not found")

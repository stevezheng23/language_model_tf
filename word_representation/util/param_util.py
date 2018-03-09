import argparse
import codecs
import json

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams"]

def create_default_hyperparams():
    """create default hyperparameters"""
    hyperparams = tf.contrib.training.HParams(
        data_log_output_dir="",
        data_result_output_dir="",
        train_ckpt_output_dir="",
        train_summary_output_dir="",
        device_num_gpus=1,
        device_default_gpu_id=0,
        device_log_device_placement=False,
        device_allow_soft_placement=False,
        device_allow_growth=False,
        device_per_process_gpu_memory_fraction=0.95
    )
    
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

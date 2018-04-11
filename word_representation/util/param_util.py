import argparse
import codecs
import json

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams"]

def create_default_hyperparams():
    """create default hyperparameters"""
    hyperparams = tf.contrib.training.HParams(
        data_train_file="",
        data_eval_file="",
        data_vocab_file="",
        data_embedding_file="",
        data_full_embedding_file="",
        data_vocab_size=30000,
        data_max_length=50,
        data_sos="<s>",
        data_eos="</s>",
        data_pad="<pad>",
        data_unk="<unk>",
        data_log_output_dir="",
        data_result_output_dir="",
        train_random_seed=100,
        train_enable_shuffle=False,
        train_batch_size=64,
        train_eval_batch_size=1,
        train_infer_batch_size=3,
        train_num_epoch=3,
        train_ckpt_output_dir="",
        train_summary_output_dir="",
        train_step_per_stat=10,
        train_step_per_ckpt=100,
        train_step_per_eval=100,
        train_clip_norm=5.0,
        train_optimizer_type="adam",
        train_optimizer_learning_rate=0.001,
        train_optimizer_decay_mode="exponential_decay",
        train_optimizer_decay_rate=0.95,
        train_optimizer_decay_step=1000,
        train_optimizer_decay_start_step=10000,
        train_optimizer_momentum_beta=0.9,
        train_optimizer_rmsprop_beta=0.999,
        train_optimizer_rmsprop_epsilon=1e-8,
        train_optimizer_adadelta_rho=0.95,
        train_optimizer_adadelta_epsilon=1e-8,
        train_optimizer_adagrad_init_accumulator=0.1,
        train_optimizer_adam_beta_1=0.9,
        train_optimizer_adam_beta_2=0.999,
        train_optimizer_adam_epsilon=1e-08,
        model_type="vanilla",
        model_scope="lm",
        model_pretrained_embedding=False,
        model_embed_dim=300,
        model_encoder_type="bi",
        model_encoder_num_layer=1,
        model_encoder_unit_dim=512,
        model_encoder_unit_type="lstm",
        model_encoder_hidden_activation="tanh",
        model_encoder_residual_connect=False,
        model_encoder_forget_bias=1.0,
        model_encoder_dropout=0.1,
        model_decoder_projection_activation="tanh",
        model_decoder_prediction_type="sample",
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

import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.train_logger import *
#from util.eval_util import *
from util.eval_logger import *
from util.debug_logger import *
from util.summary_writer import *
#from util.result_writer import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def intrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   embedding,
                   global_step):
    load_model(sess, model)
    sess.run(model.data_pipeline.initializer)
    
    loss = 0.0
    word_count = 0
    sample_size = 0
    while True:
        try:
            eval_result = model.model.evaluate(sess, embedding)
            loss += eval_result.loss * eval_result.batch_size
            word_count += eval_result.word_count
            sample_size += eval_result.batch_size
        except  tf.errors.OutOfRangeError:
            break
    
    metric = "perplexity"
    perplexity = safe_exp(loss/word_count)
    summary_writer.add_value_summary(metric, perplexity, global_step)
    intrinsic_eval_result = IntrinsicEvalLog(metric=metric,
        score=perplexity, sample_size=sample_size)
    logger.update_intrinsic_eval(intrinsic_eval_result)
    logger.check_intrinsic_eval()

def train(logger,
          hyperparams):
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    
    logger.log_print("##### start training #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    
    init_model(train_sess, train_model)
    init_model(eval_sess, eval_model)
    
    global_step = 0
    train_model.model.save(train_sess, global_step)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    for epoch in range(hyperparams.train_num_epoch):
        train_sess.run(train_model.data_pipeline.initializer)
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess, train_model.embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)

                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step)
                if step_in_epoch % hyperparams.train_step_per_eval == 0:
                    intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
                        eval_model.embedding, global_step)
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_model.model.save(train_sess, global_step)
                intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
                    eval_model.embedding, global_step)
                break

    train_summary_writer.close_writer()
    eval_summary_writer.close_writer()
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams):
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    
    logger.log_print("##### start evaluation #####")
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    
    init_model(eval_sess, eval_model)
    
    global_step = 0
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    intrinsic_eval(eval_logger, eval_summary_writer, eval_sess, eval_model,
        eval_model.embedding, global_step)
    
    eval_summary_writer.close_writer()
    logger.log_print("##### finish evaluation #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

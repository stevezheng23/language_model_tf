import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.debug_logger import *
from util.train_logger import *
from util.eval_logger import *
from util.summary_writer import *
from util.result_writer import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def intrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   input_data,
                   word_embedding,
                   batch_size,
                   global_step,
                   epoch,
                   ckpt_file,
                   eval_mode):
    data_size = len(input_data)
    load_model(sess, model, ckpt_file, eval_mode)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.data_pipeline.input_text_placeholder: input_data,
            model.data_pipeline.data_size_placeholder: data_size,
            model.data_pipeline.batch_size_placeholder: batch_size})
    
    loss = 0.0
    word_count = 0
    sample_size = 0
    while True:
        try:
            eval_result = model.model.evaluate(sess, word_embedding)
            loss += eval_result.loss * eval_result.batch_size
            word_count += eval_result.word_count
            sample_size += eval_result.batch_size
        except  tf.errors.OutOfRangeError:
            break
    
    metric = "perplexity"
    score = safe_exp(loss / word_count)
    eval_result = IntrinsicEvalLog(metric=metric, score=score, sample_size=sample_size)
    basic_info = BasicInfoEvalLog(epoch=epoch, global_step=global_step)
    summary_writer.add_value_summary(metric, score, global_step)
    
    logger.update_intrinsic_eval(eval_result, basic_info)
    logger.check_intrinsic_eval()

def sample_decode(logger,
                  sess,
                  model,
                  input_data,
                  word_embedding,
                  sample_size,
                  random_seed,
                  global_step,
                  epoch,
                  ckpt_file,
                  eval_mode):
    np.random.seed(random_seed)
    sample_index = np.random.randint(0, len(input_data), size=sample_size)
    sample_data = [input_data[index] for index in sample_index]
    
    load_model(sess, model, ckpt_file, eval_mode)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.data_pipeline.input_text_placeholder: sample_data,
            model.data_pipeline.data_size_placeholder: sample_size,
            model.data_pipeline.batch_size_placeholder: sample_size})
    
    decode_result = model.model.decode(sess, word_embedding)
    
    sample_input_list = []
    sample_output_list = []
    sample_reference_list = []
    for sample_index in range(len(sample_data)):
        sample_length = decode_result.sequence_length[sample_index]
        sample_position = np.random.randint(0, sample_length-1, size=1)[0]
        
        sample_input = sample_data[sample_index].split(' ')[:sample_length]
        sample_output = decode_result.decode_output[sample_index]
        
        sample_input_list.append(' '.join(sample_input[:sample_position] + ['(?)'] + sample_input[sample_position+1:]))
        sample_output_list.append(' '.join(sample_input[:sample_position] +
            ['({0})'.format(sample_output[sample_position].decode("utf-8"))] + sample_input[sample_position+1:]))
        sample_reference_list.append(' '.join(sample_input[:sample_position] +
            ['({0})'.format(sample_input[sample_position])] + sample_input[sample_position+1:]))
    
    sample_decode_list = [{
        "sample_input": sample_input,
        "sample_output": sample_output,
        "sample_reference": sample_reference
    } for sample_input, sample_output, sample_reference in list(zip(sample_input_list, sample_output_list, sample_reference_list))]
    
    decode_result = DecodeEvalLog(sample_decode_list=sample_decode_list)
    basic_info = BasicInfoEvalLog(epoch=epoch, global_step=global_step)
    
    logger.update_sample_decode(decode_result, basic_info)
    logger.check_sample_decode()

def sample_encode(result_writer,
                  sess,
                  model,
                  input_data,
                  word_embedding,
                  encode_type,
                  encode_layer_list,
                  batch_size,
                  global_step,
                  epoch,
                  ckpt_file,
                  eval_mode):
    data_size = len(input_data)
    load_model(sess, model, ckpt_file, eval_mode)
    sess.run(model.data_pipeline.initializer,
        feed_dict={model.data_pipeline.input_text_placeholder: input_data,
            model.data_pipeline.data_size_placeholder: data_size,
            model.data_pipeline.batch_size_placeholder: batch_size})
    
    encode_result_list = []
    while True:
        try:
            encode_result = model.model.encode(sess, word_embedding)
            encode_result_batch = [{
                "sample_encode": list(encode_result.encode_output[i].tolist()),
                "sequence_length": int(encode_result.sequence_length[i]),
                "encode_type": encode_type,
                "encode_layer_list": encode_layer_list
            } for i in range(encode_result.batch_size)]
            
            encode_result_list.extend(encode_result_batch)
        except  tf.errors.OutOfRangeError:
            break
    
    if data_size != len(encode_result_list):
        raise ValueError("encode result size is not equal to input data size")
    
    result_writer.write_result(encode_result_list, "encode", "{0}_{1}".format(global_step, epoch))

def train(logger,
          hyperparams,
          enable_eval=True,
          enable_debug=False):
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    if enable_debug == True:
        train_sess = tf_debug.LocalCLIDebugWrapperSession(train_sess)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    init_model(train_sess, train_model)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    
    if enable_eval == True:
        logger.log_print("##### create eval model #####")
        eval_model = create_eval_model(logger, hyperparams)
        eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
        
        logger.log_print("##### create decode model #####")
        decode_model = create_decode_model(logger, hyperparams)
        decode_sess = tf.Session(config=config_proto, graph=decode_model.graph)
        
        if enable_debug == True:
            eval_sess = tf_debug.LocalCLIDebugWrapperSession(eval_sess)
            decode_sess = tf_debug.LocalCLIDebugWrapperSession(decode_sess)
        
        eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
        decode_summary_writer = SummaryWriter(decode_model.graph, os.path.join(summary_output_dir, "decode"))
        
        init_model(eval_sess, eval_model)
        init_model(decode_sess, decode_model)
        
        eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start training #####")
    global_step = 0
    train_model.model.save(train_sess, global_step, "debug")
    for epoch in range(hyperparams.train_num_epoch):
        train_sess.run(train_model.data_pipeline.initializer)
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess, train_model.word_embedding)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)
                
                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step, "debug")
                if step_in_epoch % hyperparams.train_step_per_eval == 0 and enable_eval == True:
                    ckpt_file = eval_model.model.get_latest_ckpt("debug")
                    intrinsic_eval(eval_logger, eval_summary_writer,
                        eval_sess, eval_model, eval_model.input_data, eval_model.word_embedding,
                        hyperparams.train_eval_batch_size, global_step, epoch, ckpt_file, "debug")
                    sample_decode(eval_logger, decode_sess, decode_model, decode_model.input_data,
                        decode_model.word_embedding, hyperparams.train_decode_sample_size,
                        hyperparams.train_random_seed + global_step, global_step, epoch, ckpt_file, "debug")
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_summary_writer.add_summary(train_result.summary, global_step)
                train_model.model.save(train_sess, global_step, "epoch")
                if enable_eval == True:
                    ckpt_file = eval_model.model.get_latest_ckpt("epoch")
                    intrinsic_eval(eval_logger, eval_summary_writer,
                        eval_sess, eval_model, eval_model.input_data, eval_model.word_embedding,
                        hyperparams.train_eval_batch_size, global_step, epoch, ckpt_file, "epoch")
                    sample_decode(eval_logger, decode_sess, decode_model, decode_model.input_data,
                        decode_model.word_embedding, hyperparams.train_decode_sample_size,
                        hyperparams.train_random_seed + global_step, global_step, epoch, ckpt_file, "epoch")
                break
    
    train_summary_writer.close_writer()
    if enable_eval == True:
        eval_summary_writer.close_writer()
    
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams,
             enable_debug=False):   
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create eval model #####")
    eval_model = create_eval_model(logger, hyperparams)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)

    logger.log_print("##### create decode model #####")
    decode_model = create_decode_model(logger, hyperparams)
    decode_sess = tf.Session(config=config_proto, graph=decode_model.graph)
    
    if enable_debug == True:
        eval_sess = tf_debug.LocalCLIDebugWrapperSession(eval_sess)
        decode_sess = tf_debug.LocalCLIDebugWrapperSession(decode_sess)
    
    eval_summary_writer = SummaryWriter(eval_model.graph, os.path.join(summary_output_dir, "eval"))
    
    init_model(eval_sess, eval_model)
    init_model(decode_sess, decode_model)
    
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start evaluation #####")
    eval_mode = "debug" if enable_debug == True else "epoch"
    ckpt_file_list = eval_model.model.get_ckpt_list(eval_mode)
    for i, ckpt_file in enumerate(ckpt_file_list):
        intrinsic_eval(eval_logger, eval_summary_writer,
            eval_sess, eval_model, eval_model.input_data, eval_model.word_embedding,
            hyperparams.train_eval_batch_size, i, i, ckpt_file, eval_mode)
        sample_decode(eval_logger, decode_sess, decode_model, decode_model.input_data,
            decode_model.word_embedding, hyperparams.train_decode_sample_size,
            hyperparams.train_random_seed + i, i, i, ckpt_file, eval_mode)
    
    eval_summary_writer.close_writer()
    logger.log_print("##### finish evaluation #####")

def encode(logger,
           hyperparams,
           enable_debug=False):
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create encode model #####")
    encode_model = create_encode_model(logger, hyperparams)
    encode_sess = tf.Session(config=config_proto, graph=encode_model.graph)
    
    if enable_debug == True:
        encode_sess = tf_debug.LocalCLIDebugWrapperSession(encode_sess)
    
    init_model(encode_sess, encode_model)
    
    result_writer = ResultWriter(hyperparams.data_result_output_dir)
    
    logger.log_print("##### start encoding #####")
    encode_mode = "debug" if enable_debug == True else "epoch"
    ckpt_file = encode_model.model.get_latest_ckpt(encode_mode)
    sample_encode(result_writer, encode_sess, encode_model, encode_model.input_data,
        encode_model.word_embedding, hyperparams.model_encode_type, hyperparams.model_encode_layer_list,
        hyperparams.train_encode_batch_size, 0, 0, ckpt_file, encode_mode)
    
    logger.log_print("##### finish encoding #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train_eval'):
        train(logger, hyperparams, enable_eval=True, enable_debug=False)
    elif (args.mode == 'train'):
        train(logger, hyperparams, enable_eval=False, enable_debug=False)
    elif (args.mode == 'train_debug'):
        train(logger, hyperparams, enable_eval=False, enable_debug=True)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams, enable_debug=False)
    elif (args.mode == 'eval_debug'):
        evaluate(logger, hyperparams, enable_debug=True)
    elif (args.mode == 'encode'):
        encode(logger, hyperparams, enable_debug=False)
    elif (args.mode == 'encode_debug'):
        encode(logger, hyperparams, enable_debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

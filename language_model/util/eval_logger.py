import codecs
import collections
import os.path
import time

import numpy as np
import tensorflow as tf

__all__ = ["IntrinsicEvalLog", "DecodeEvalLog", "EvalLogger"]

class IntrinsicEvalLog(collections.namedtuple("IntrinsicEvalLog", ("metric", "score", "sample_size"))):
    pass

class DecodeEvalLog(collections.namedtuple("DecodeEvalLog", ("sample_decode_list"))):
    pass

class EvalLogger(object):
    """eval logger"""    
    def __init__(self,
                 output_dir):
        """intrinsic eval result"""
        self.intrinsic_eval = None
        self.intrinsic_eval_info = None
        
        """sample decode result"""
        self.sample_decode = None
        self.sample_decode_info = None
        
        """initialize eval logger"""        
        self.output_dir = output_dir
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)
        self.log_file = os.path.join(self.output_dir, "eval_{0}.log".format(time.time()))
        self.log_writer = codecs.getwriter("utf-8")(tf.gfile.GFile(self.log_file, mode="a"))
    
    def update_intrinsic_eval(self,
                              eval_result,
                              basic_info):
        """update eval logger with intrinsic eval result"""
        self.intrinsic_eval = eval_result
        self.intrinsic_eval_info = basic_info
    
    def update_sample_decode(self,
                             decode_result,
                             basic_info):
        """update eval logger with sample decode result"""
        self.sample_decode = decode_result
        self.sample_decode_info = basic_info
    
    def check_intrinsic_eval(self):
        """check intrinsic eval result"""
        log_line = "epoch={0}, global step={1}, {2}={3}, sample size={4}".format(
            self.intrinsic_eval_info.epoch, self.intrinsic_eval_info.global_step, self.intrinsic_eval.metric,
            self.intrinsic_eval.score, self.intrinsic_eval.sample_size).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
    
    def check_sample_decode(self):
        """check sample decode result"""
        sample_size = len(self.sample_decode.sample_decode_list)
        log_line = "epoch={0}, global step={1}, sample size={2}".format(self.sample_decode_info.epoch,
            self.sample_decode_info.global_step, sample_size).encode('utf-8')
        self.log_writer.write("{0}\r\n".format(log_line))
        print(log_line)
        
        for i, sample_decode in enumerate(self.sample_decode.sample_decode_list):
            sample_input = sample_decode["sample_input"]
            log_line = "sample {0} - input: {1}".format(i+1, sample_input).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            sample_output = sample_decode["sample_output"]
            log_line = "sample {0} - output: {1}".format(i+1, sample_output).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)
            sample_reference = sample_decode["sample_reference"]
            log_line = "sample {0} - reference: {1}".format(i+1, sample_reference).encode('utf-8')
            self.log_writer.write("{0}\r\n".format(log_line))
            print(log_line)

import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.language_model_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["SequenceLM"]

class SequenceLM(BaseModel):
    """sequence language model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="seq_lm"):
        """initialize sequence language model"""
        super(SequenceLM, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            """get batch input"""
            text_word = self.data_pipeline.input_text_word
            text_word_mask = self.data_pipeline.input_text_word_mask
            text_char = self.data_pipeline.input_text_char
            text_char_mask = self.data_pipeline.input_text_char_mask
            self.word_vocab_invert_index = self.data_pipeline.word_vocab_inverted_index
            self.word_vocab_size = self.data_pipeline.word_vocab_size
            self.char_vocab_size = self.data_pipeline.char_vocab_size
            self.sequence_length = tf.cast(tf.reduce_sum(text_word_mask, axis=[-1, -2]), dtype=tf.int32)
            
            """build graph"""
            if self.mode in ["train", "eval", "decode"]:
                self.logger.log_print("# build graph")
                predict, predict_mask = self._build_graph(text_word, text_word_mask, text_char, text_char_mask)
                
                label = tf.cast(text_word, dtype=tf.float32)
                label_mask = text_word_mask
                label, label_mask = align_sequence(label, label_mask, 1)
                label, label_mask = reverse_sequence(label, label_mask)
                label, label_mask = align_sequence(label, label_mask, 1)
                label, label_mask = reverse_sequence(label, label_mask)
            
            """build encode graph"""
            if self.mode == "encode":
                self.logger.log_print("# build encode graph")
                result, result_mask = self._build_encode_graph(text_word, text_word_mask, text_char, text_char_mask)
                self.encode_result = result
                self.encode_sequence_length = tf.cast(tf.reduce_sum(result_mask, axis=[-1, -2]), dtype=tf.int32)
            
            """compute loss"""
            if self.mode in ["train", "eval"]:
                self.logger.log_print("# setup loss computation mechanism")
                loss = self._compute_loss(label, label_mask, predict, predict_mask)
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    loss = loss + regularization_loss
                
                self.train_loss = loss
                self.eval_loss = loss
                self.word_count = tf.reduce_sum(predict_mask)
            
            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.variable_lookup = {v.op.name: v for v in self.variable_list}
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = tf.train.ExponentialMovingAverage(decay=self.hyperparams.train_ema_decay_rate)
                self.variable_lookup = {self.ema.average_name(v): v for v in self.variable_list}
            
            """decode output"""
            if self.mode == "decode":
                softmax_predict = softmax_with_mask(predict, predict_mask, axis=-1)
                index_predict = tf.argmax(softmax_predict, axis=-1, output_type=tf.int64)
                self.decode_predict = self.word_vocab_invert_index.lookup(index_predict)
                self.decode_sequence_length = tf.cast(tf.reduce_sum(predict_mask, axis=[-1, -2]), dtype=tf.int32)
            
            """apply training"""
            if self.mode == "train":
                self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                    initializer=tf.zeros_initializer, trainable=False)
                
                self.logger.log_print("# setup initial learning rate mechanism")
                self.initial_learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_warmup_enable == True:
                    self.logger.log_print("# setup learning rate warm-up mechanism")
                    self.warmup_learning_rate = self._apply_learning_rate_warmup(self.initial_learning_rate)
                else:
                    self.warmup_learning_rate = self.initial_learning_rate
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.warmup_learning_rate)
                else:
                    self.decayed_learning_rate = self.warmup_learning_rate
                
                self.learning_rate = self.decayed_learning_rate
                
                self.logger.log_print("# setup training optimizer")
                self.optimizer = self._initialize_optimizer(self.learning_rate)
                
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.update_model]):
                        self.update_op = self.ema.apply(self.variable_list)
                        self.variable_lookup = {self.ema.average_name(v): self.ema.average(v) for v in self.variable_list}
                else:
                    self.update_op = self.update_model
                
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_debug_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "debug")
            self.ckpt_epoch_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "epoch")
            
            if not tf.gfile.Exists(self.ckpt_debug_dir):
                tf.gfile.MakeDirs(self.ckpt_debug_dir)
            
            if not tf.gfile.Exists(self.ckpt_epoch_dir):
                tf.gfile.MakeDirs(self.ckpt_epoch_dir)
            
            self.ckpt_debug_name = os.path.join(self.ckpt_debug_dir, "model_debug_ckpt")
            self.ckpt_epoch_name = os.path.join(self.ckpt_epoch_dir, "model_epoch_ckpt")
            self.ckpt_debug_saver = tf.train.Saver(self.variable_lookup)
            self.ckpt_epoch_saver = tf.train.Saver(self.variable_lookup, max_to_keep=self.hyperparams.train_num_epoch)
    
    def _build_representation_layer(self,
                                    text_word,
                                    text_word_mask,
                                    text_char,
                                    text_char_mask):
        """build representation layer for sequence language model"""
        word_embed_dim = self.hyperparams.model_word_embed_dim
        word_dropout = self.hyperparams.model_word_dropout if self.mode == "train" else 0.0
        word_embed_pretrained = self.hyperparams.model_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_word_feat_trainable
        word_feat_enable = self.hyperparams.model_word_feat_enable
        char_embed_dim = self.hyperparams.model_char_embed_dim
        char_unit_dim = self.hyperparams.model_char_unit_dim
        char_window_size = self.hyperparams.model_char_window_size
        char_hidden_activation = self.hyperparams.model_char_hidden_activation
        char_dropout = self.hyperparams.model_char_dropout if self.mode == "train" else 0.0
        char_pooling_type = self.hyperparams.model_char_pooling_type
        char_feat_trainable = self.hyperparams.model_char_feat_trainable
        char_feat_enable = self.hyperparams.model_char_feat_enable
        fusion_type = self.hyperparams.model_fusion_type
        fusion_num_layer = self.hyperparams.model_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_fusion_trainable
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            text_feat_list = []
            text_feat_mask_list = []
            
            if word_feat_enable == True:
                self.logger.log_print("# build word-level representation layer")
                word_feat_layer = WordFeat(vocab_size=self.word_vocab_size, embed_dim=word_embed_dim,
                    dropout=word_dropout, pretrained=word_embed_pretrained, random_seed=self.random_seed,
                    feedable=True, trainable=word_feat_trainable)
                
                (text_word_feat,
                    text_word_feat_mask) = word_feat_layer(text_word, text_word_mask)
                text_feat_list.append(text_word_feat)
                text_feat_mask_list.append(text_word_feat_mask)
                
                word_unit_dim = word_embed_dim
                self.word_embedding_placeholder = word_feat_layer.get_embedding_placeholder()
            else:
                word_unit_dim = 0
                self.word_embedding_placeholder = None
            
            if char_feat_enable == True:
                self.logger.log_print("# build char-level representation layer")
                char_feat_layer = CharFeat(vocab_size=self.char_vocab_size, embed_dim=char_embed_dim, unit_dim=char_unit_dim,
                    window_size=char_window_size, activation=char_hidden_activation, pooling_type=char_pooling_type,
                    dropout=char_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=char_feat_trainable)
                
                (text_char_feat,
                    text_char_feat_mask) = char_feat_layer(text_char, text_char_mask)
                
                text_feat_list.append(text_char_feat)
                text_feat_mask_list.append(text_char_feat_mask)
            else:
                char_unit_dim = 0
            
            feat_unit_dim = word_unit_dim + char_unit_dim
            feat_fusion_layer = FusionModule(input_unit_dim=feat_unit_dim, output_unit_dim=fusion_unit_dim,
                fusion_type=fusion_type, num_layer=fusion_num_layer, activation=fusion_hidden_activation,
                dropout=fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                regularizer=self.regularizer, random_seed=self.random_seed, trainable=fusion_trainable)
            
            text_feat, text_feat_mask = feat_fusion_layer(text_feat_list, text_feat_mask_list)
        
        return text_feat, text_feat_mask
    
    def _build_modeling_layer(self,
                              text_feat,
                              text_feat_mask):
        """build modeling layer for sequence language model"""
        sequence_num_layer = self.hyperparams.model_sequence_num_layer
        sequence_unit_dim = self.hyperparams.model_sequence_unit_dim
        sequence_cell_type = self.hyperparams.model_sequence_cell_type
        sequence_hidden_activation = self.hyperparams.model_sequence_hidden_activation
        sequence_dropout = self.hyperparams.model_sequence_dropout if self.mode == "train" else 0.0
        sequence_forget_bias = self.hyperparams.model_sequence_forget_bias
        sequence_residual_connect = self.hyperparams.model_sequence_residual_connect
        sequence_trainable = self.hyperparams.model_sequence_trainable
        projection_dropout = self.hyperparams.model_projection_dropout
        projection_trainable = self.hyperparams.model_projection_trainable
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build sequence modeling layer")
            sequence_layer = create_recurrent_layer("stacked_bi", sequence_num_layer, sequence_unit_dim,
                sequence_cell_type, sequence_hidden_activation, sequence_dropout, sequence_forget_bias,
                sequence_residual_connect, None, self.num_gpus, self.default_gpu_id, self.random_seed, sequence_trainable)
            
            (text_sequence_list,
                text_sequence_mask_list) = sequence_layer(text_feat, text_feat_mask)
            text_modeling_list = text_sequence_list
            text_modeling_mask_list = text_sequence_mask_list
        
        return text_modeling_list, text_modeling_mask_list
    
    def _build_output_layer(self,
                            text_modeling_list,
                            text_modeling_mask_list):
        """build output layer for sequence language model"""
        projection_dropout = self.hyperparams.model_projection_dropout
        projection_trainable = self.hyperparams.model_projection_trainable
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            text_modeling = text_modeling_list[-1]
            text_modeling_mask = text_modeling_mask_list[-1]
            
            projection_layer = create_dense_layer("single", 1, self.word_vocab_size, 1, "", [projection_dropout], None,
                False, False, False, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, projection_trainable)
            
            (text_projection,
                text_projection_mask) = projection_layer(text_modeling, text_modeling_mask)
            
            text_output = text_projection
            text_output_mask = text_projection_mask
        
        return text_output, text_output_mask
    
    def _build_encode_layer(self,
                            text_modeling_list,
                            text_modeling_mask_list):
        """build encode layer for sequence language model"""
        encode_type = self.hyperparams.model_encode_type
        encode_layer_list = self.hyperparams.model_encode_layer_list
        
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            text_encode = text_modeling_list[-1]
            text_encode_mask = text_modeling_mask_list[-1]
        
        return text_encode, text_encode_mask
    
    def _build_graph(self,
                     text_word,
                     text_word_mask,
                     text_char,
                     text_char_mask):
        """build graph for sequence language model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for sequence language model"""
            text_feat, text_feat_mask = self._build_representation_layer(text_word,
                text_word_mask, text_char, text_char_mask)
            
            """build modeling layer for sequence language model"""
            text_modeling_list, text_modeling_mask_list = self._build_modeling_layer(text_feat, text_feat_mask)
            
            """build output layer for sequence language model"""
            text_output, text_output_mask = self._build_output_layer(text_modeling_list, text_modeling_mask_list)
            
            predict = text_output
            predict_mask = text_output_mask
        
        return predict, predict_mask
    
    def _build_encode_graph(self,
                            text_word,
                            text_word_mask,
                            text_char,
                            text_char_mask):
        """build encode graph for sequence language model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for sequence language model"""
            text_feat, text_feat_mask = self._build_representation_layer(text_word,
                text_word_mask, text_char, text_char_mask)
            
            """build modeling layer for sequence language model"""
            text_modeling_list, text_modeling_mask_list = self._build_modeling_layer(text_feat, text_feat_mask)
            
            """build encoding layer for sequence language model"""
            text_encode, text_encode_mask = self._build_encode_layer(text_modeling_list, text_modeling_mask_list)
            
            result = text_encode
            result_mask = text_encode_mask
        
        return result, result_mask
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask):
        """compute optimization loss"""
        masked_predict = generate_masked_data(predict, predict_mask)
        masked_label = tf.cast(label * label_mask, dtype=tf.int32)
        onehot_label = generate_onehot_label(masked_label, tf.shape(predict)[-1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_predict, labels=onehot_label)
        loss = tf.reduce_sum(cross_entropy * tf.squeeze(predict_mask, axis=-1)) / tf.cast(self.batch_size, dtype=tf.float32)
        return loss
    
    def save(self,
             sess,
             global_step,
             save_mode):
        """save checkpoint for sequence language model"""
        if save_mode == "debug":
            self.ckpt_debug_saver.save(sess, self.ckpt_debug_name, global_step=global_step)
        elif save_mode == "epoch":
            self.ckpt_epoch_saver.save(sess, self.ckpt_epoch_name, global_step=global_step)
        else:
            raise ValueError("unsupported save mode {0}".format(save_mode))
    
    def restore(self,
                sess,
                ckpt_file,
                ckpt_type):
        """restore sequence language model from checkpoint"""
        if ckpt_file is None:
            raise FileNotFoundError("checkpoint file doesn't exist")
        
        if ckpt_type == "debug":
            self.ckpt_debug_saver.restore(sess, ckpt_file)
        elif ckpt_type == "epoch":
            self.ckpt_epoch_saver.restore(sess, ckpt_file)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_latest_ckpt(self,
                        ckpt_type):
        """get the latest checkpoint for sequence language model"""
        if ckpt_type == "debug":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_debug_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        elif ckpt_type == "epoch":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_epoch_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_ckpt_list(self,
                      ckpt_type):
        """get checkpoint list for sequence language model"""
        if ckpt_type == "debug":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_debug_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        elif ckpt_type == "epoch":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_epoch_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 dropout,
                 pretrained,
                 random_seed=0,
                 feedable=True,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.random_seed = random_seed
        self.feedable = feedable
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, self.pretrained, 0, 0, self.random_seed, self.feedable, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, 0, 0, self.random_seed)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_embedding_mask = input_word_mask
            input_word_embedding = tf.squeeze(self.embedding_layer(input_word), axis=-2)
            
            (input_word_dropout,
                input_word_dropout_mask) = self.dropout_layer(input_word_embedding, input_word_embedding_mask)
            
            input_word_feat = input_word_dropout
            input_word_feat_mask = input_word_dropout_mask
        
        return input_word_feat, input_word_feat_mask
    
    def get_embedding_placeholder(self):
        """get word-level embedding placeholder"""
        return self.embedding_layer.get_embedding_placeholder()

class CharFeat(object):
    """char-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="char_feat"):
        """initialize char-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, False, 0, 0, self.random_seed, False, self.trainable)
            
            self.conv_layer = create_convolution_layer("stacked_multi_1d", 1, self.embed_dim,
                self.unit_dim, self.window_size, 1, "SAME", self.activation, [self.dropout], None,
                False, False, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, -1, 1, 0, 0)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            input_char_embedding = self.embedding_layer(input_char)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_embedding, input_char_embedding_mask)
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask

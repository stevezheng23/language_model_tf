import collections
import os.path

import numpy as np
import tensorflow as tf

from util.representation_util import *

__all__ = ["TrainResult", "EvaluateResult", "LanguageModel"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult", ("loss", "batch_size", "word_count"))):
    pass

class LanguageModel(object):
    """language model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 vocab_size,
                 vocab_index,
                 vocab_inverted_index,
                 mode="train",
                 pretrained_embedding=False,
                 scope="lm"):
        """initialize language model"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.logger = logger
            self.hyperparams = hyperparams
            
            self.data_pipeline = data_pipeline
            self.vocab_size = vocab_size
            self.vocab_index = vocab_index
            self.vocab_inverted_index = vocab_inverted_index
            self.mode = mode
            self.pretrained_embedding = pretrained_embedding
            self.scope = scope
            
            self.num_gpus = self.hyperparams.device_num_gpus
            self.default_gpu_id = self.hyperparams.device_default_gpu_id
            self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
                .format(self.num_gpus, self.default_gpu_id))
            
            """get batch inputs from data pipeline"""
            text_input = self.data_pipeline.text_input
            text_input_length = self.data_pipeline.text_input_length
            self.batch_size = tf.size(text_input_length)
            
            """build graph for language model"""
            self.logger.log_print("# build graph for language model")
            (logit, encoder_output, encoder_final_state,
                input_embedding, embedding_placeholder) = self._build_graph(text_input, text_input_length)
            self.input_embedding = input_embedding
            self.embedding_placeholder = embedding_placeholder
            
            if self.mode == "train" or self.mode == "eval":
                logit_length = self.data_pipeline.text_output_length
                self.word_count = tf.reduce_sum(logit_length)
                
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                label = self.data_pipeline.text_output
                loss = self._compute_loss(logit, label, logit_length)
                self.train_loss = loss
                self.eval_loss = loss
                
                """apply learning rate decay"""
                self.logger.log_print("# setup learning rate decay mechanism")
                self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                    initializer=tf.zeros_initializer, trainable=False)
                self.learning_rate = tf.get_variable("learning_rate", dtype=tf.float32,
                    initializer=tf.constant(self.hyperparams.train_optimizer_learning_rate), trainable=False)
                decayed_learning_rate = self._apply_learning_rate_decay(self.learning_rate)
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                """create summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_embedding(self,
                         input_data):
        """build embedding layer for language model"""
        embed_dim = self.hyperparams.model_embed_dim
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create embedding for language model")
            embedding, embedding_placeholder = create_embedding(self.vocab_size,
                embed_dim, self.pretrained_embedding)
            input_embedding = tf.nn.embedding_lookup(embedding, input_data)
            
            return input_embedding, embedding_placeholder
    
    def _create_encoder_cell(self,
                             num_layer,
                             unit_dim,
                             unit_type,
                             activation,
                             forget_bias,
                             residual_connect,
                             drop_out):
        """create encoder cell"""
        cell = create_rnn_cell(num_layer, unit_dim, unit_type, activation,
            forget_bias, residual_connect, drop_out, self.num_gpus, self.default_gpu_id)
        
        return cell
    
    def _convert_encoder_outputs(self,
                                 outputs):
        """convert encoder outputs"""
        encoder_type = self.hyperparams.model_encoder_type
        if encoder_type == "bi":
            outputs = tf.concat(outputs, -1)
        
        return outputs
    
    def _convert_encoder_state(self,
                               state):
        """convert encoder state"""
        encoder_type = self.hyperparams.model_encoder_type
        num_layer = self.hyperparams.model_encoder_num_layer
        if encoder_type == "bi":
            if num_layer > 1:
                state_list = []
                for i in range(num_layer):
                    state_list.append(state[0][i])
                    state_list.append(state[1][i])
                state = tuple(state_list)
        
        return state
    
    def _build_encoder(self,
                       encoder_input,
                       encoder_input_length):
        """build encoder layer for language model"""
        encoder_type = self.hyperparams.model_encoder_type
        num_layer = self.hyperparams.model_encoder_num_layer
        unit_dim = self.hyperparams.model_encoder_unit_dim
        unit_type = self.hyperparams.model_encoder_unit_type
        hidden_activation = self.hyperparams.model_encoder_hidden_activation
        residual_connect = self.hyperparams.model_encoder_residual_connect
        forget_bias = self.hyperparams.model_encoder_forget_bias
        drop_out = self.hyperparams.model_encoder_dropout
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create hidden layer for encoder")
            if encoder_type == "uni":
                cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                encoder_output, encoder_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_input,
                    sequence_length=encoder_input_length, dtype=tf.float32)
            elif encoder_type == "bi":
                fwd_cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                bwd_cell = self._create_encoder_cell(num_layer, unit_dim, unit_type, hidden_activation,
                    forget_bias, residual_connect, drop_out)
                encoder_output, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell,
                    inputs=encoder_input, sequence_length=encoder_input_length, dtype=tf.float32)
            else:
                raise ValueError("unsupported encoder type {0}".format(encoder_type))
            
            encoder_output = self._convert_encoder_outputs(encoder_output)
            encoder_final_state = self._convert_encoder_state(encoder_final_state)
                        
            return encoder_output, encoder_final_state
        
    def _build_decoder(self,
                       decoder_input):
        """build decoder layer for language model"""
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            projection_activation = create_activation_function(self.hyperparams.model_projection_activation)
            
            """create projection layer for decoder"""
            self.logger.log_print("# create projection layer for decoder")
            projector = tf.layers.Dense(units=self.vocab_size, activation=projection_activation)
            
            decoder_output = projector.apply(decoder_input)
            return decoder_output
    
    def _build_graph(self,
                     input_data,
                     input_length):
        """build graph for language model"""       
        self.logger.log_print("# build embedding layer for language model")
        input_embedding, embedding_placeholder = self._build_embedding(input_data)
        
        self.logger.log_print("# build encoder layer for language model")
        encoder_output, encoder_final_state = self._build_encoder(input_embedding, input_length)
        
        self.logger.log_print("# build decoder layer for language model")
        decoder_output = self._build_decoder(encoder_output)
        
        return decoder_output, encoder_output, encoder_final_state, input_embedding, embedding_placeholder
    
    def _compute_loss(self,
                      logit,
                      label,
                      logit_length):
        """compute optimization loss"""
        mask = tf.sequence_mask(logit_length, maxlen=tf.shape(logit)[1], dtype=logit.dtype)
        cross_entropy = tf.contrib.seq2seq.sequence_loss(logits=logit, targets=label,
            weights=mask, average_across_timesteps=False, average_across_batch=True)
        loss = tf.reduce_sum(cross_entropy)
        
        return loss
    
    def _apply_learning_rate_decay(self,
                                   learning_rate):
        """apply learning rate decay"""
        decay_mode = self.hyperparams.train_optimizer_decay_mode
        decay_rate = self.hyperparams.train_optimizer_decay_rate
        decay_step = self.hyperparams.train_optimizer_decay_step
        decay_start_step = self.hyperparams.train_optimizer_decay_start_step
        
        if decay_mode == "exponential_decay":
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        elif decay_mode == "inverse_time_decay":
            decayed_learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        else:
            raise ValueError("unsupported decay mode {0}".format(decay_mode))
        
        decayed_learning_rate = tf.cond(tf.less(self.global_step, decay_start_step),
            lambda: learning_rate, lambda: decayed_learning_rate)
        
        return decayed_learning_rate
    
    def _initialize_optimizer(self,
                              learning_rate):
        """initialize optimizer"""
        optimizer_type = self.hyperparams.train_optimizer_type
        if optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                momentum=self.hyperparams.train_optimizer_momentum_beta)
        elif optimizer_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                decay=self.hyperparams.train_optimizer_rmsprop_beta,
                epsilon=self.hyperparams.train_optimizer_rmsprop_epsilon)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                rho=self.hyperparams.train_optimizer_adadelta_rho,
                epsilon=self.hyperparams.train_optimizer_adadelta_epsilon)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                initial_accumulator_value=self.hyperparams.train_optimizer_adagrad_init_accumulator)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                beta1=self.hyperparams.train_optimizer_adam_beta_1, beta2=self.hyperparams.train_optimizer_adam_beta_2,
                epsilon=self.hyperparams.train_optimizer_adam_epsilon)
        else:
            raise ValueError("unsupported optimizer type {0}".format(optimizer_type))
        
        return optimizer
    
    def _minimize_loss(self,
                       loss):
        """minimize optimization loss"""
        """compute gradients"""
        grads_and_vars = self.optimizer.compute_gradients(loss)
        
        """clip gradients"""
        gradients = [x[0] for x in grads_and_vars]
        variables = [x[1] for x in grads_and_vars]
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyperparams.train_clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        
        """update model based on gradients"""
        update_model = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        return update_model, clipped_gradients, gradient_norm
    
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def train(self,
              sess,
              embedding):
        """train language model"""
        if self.pretrained_embedding == True:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary])
        
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def evaluate(self,
                 sess,
                 embedding):
        """evaluate language model"""
        if self.pretrained_embedding == True:
            loss, batch_size, word_count = sess.run([self.eval_loss, self.batch_size, self.word_count],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            loss, batch_size, word_count = sess.run([self.eval_loss, self.batch_size, self.word_count])
        
        return EvaluateResult(loss=loss, batch_size=batch_size, word_count=word_count)
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for language model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore language model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")

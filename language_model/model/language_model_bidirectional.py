import numpy as np
import tensorflow as tf

from model.language_model import *

__all__ = ["LanguageModelBidirectional"]

class LanguageModelBidirectional(LanguageModel):
    """bi-directional language model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 vocab_size,
                 vocab_index,
                 vocab_inverted_index,
                 mode="train",
                 scope="bilm"):
        """initialize bi-directional language model"""
        super(LanguageModelBidirectional, self).__init__(logger=logger,
            hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index,
            vocab_inverted_index=vocab_inverted_index, mode=mode, scope=scope)
    
    def _convert_layer_input(self,
                             layer_input,
                             layer_input_length):
        """convert encoder input for bi-directional language model"""
        """convert encoder input to forward input"""
        fwd_layer_input = tf.reverse_sequence(layer_input, layer_input_length, seq_axis=1)
        fwd_layer_input = fwd_layer_input[:,1:,:]
        fwd_layer_input_length = layer_input_length - 1
        fwd_layer_input = tf.reverse_sequence(fwd_layer_input, fwd_layer_input_length, seq_axis=1)
        
        """convert encoder input to backward input"""
        bwd_layer_input = layer_input[:,1:,:]
        bwd_layer_input_length = layer_input_length - 1
        bwd_layer_input = tf.reverse_sequence(bwd_layer_input, bwd_layer_input_length, seq_axis=1)
        
        return fwd_layer_input, bwd_layer_input, fwd_layer_input_length, bwd_layer_input_length
    
    def _convert_layer_output(self,
                              fwd_layer_output,
                              bwd_layer_output,
                              fwd_layer_output_length,
                              bwd_layer_output_length):
        """convert encoder output for bi-directional langauge model"""
        padding = tf.constant([[0,0],[1,0],[0,0]])
        fwd_layer_output = tf.pad(fwd_layer_output, padding, 'CONSTANT')
        fwd_layer_output_length = fwd_layer_output_length + 1
        bwd_layer_output = tf.pad(bwd_layer_output, padding, 'CONSTANT')
        bwd_layer_output_length = bwd_layer_output_length + 1
        bwd_layer_output = tf.reverse_sequence(bwd_layer_output, bwd_layer_output_length, seq_axis=1)
        layer_output = tf.concat([fwd_layer_output, bwd_layer_output], -1)
        
        return layer_output
    
    def _build_encoder(self,
                       encoder_input,
                       encoder_input_length):
        """build encoder layer for bi-directional language model"""
        num_layer = self.hyperparams.model_encoder_num_layer
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create hidden layer for encoder")
            """convert layer input for encoder"""
            (fwd_layer_input, bwd_layer_input, fwd_layer_input_length,
                bwd_layer_input_length) = self._convert_layer_input(encoder_input, encoder_input_length)
            
            encoder_layer_output = []
            encoder_layer_final_state = []
            for i in range(num_layer):
                """build forward layer for encoder"""
                fwd_layer_output, fwd_layer_final_state = self._build_rnn_layer(fwd_layer_input,
                    fwd_layer_input_length, i, "forward")
                fwd_layer_input = fwd_layer_output
                
                """build backward layer for encoder"""
                bwd_layer_output, bwd_layer_final_state = self._build_rnn_layer(bwd_layer_input,
                    bwd_layer_input_length, i, "backward")
                bwd_layer_input = bwd_layer_output
                
                """convert layer output & state for encoder"""
                layer_output = self._convert_layer_output(fwd_layer_output, bwd_layer_output,
                    fwd_layer_input_length, bwd_layer_input_length)
                layer_final_state = tf.concat([fwd_layer_final_state, bwd_layer_final_state], -1)
                encoder_layer_output.append(layer_output)
                encoder_layer_final_state.append(layer_final_state)
            
            return encoder_layer_output, encoder_layer_final_state

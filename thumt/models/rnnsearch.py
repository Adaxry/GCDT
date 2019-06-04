# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers
from thumt.utils.utils import load_glove
from six.moves import reduce
from tensorflow.python.ops import variable_scope as vs


def get_rnn_cell(rnn_type, hidden_size, rnn_dropout, transition_num=4):
    """ get a rnn_cell
    Parameters
    ----------
    rnn_type : str
        rnn type between "GRU" and "DT"
    hidden_size : int
        hidden size of rnn cell
    rnn_dropout : float
        dropput rate of rnn cell
    transition_num : int
        transition num of "DT", default is 4
    """
    cell = None
    if rnn_type == "DT":
        cell = layers.rnn_cell.DL4MTGRULAUTransiLNCell(transition_num, hidden_size, 1.0 - rnn_dropout)

    elif rnn_type == "GRU":
        cell = layers.rnn_cell.LegacyGRUCell(hidden_size)

    else:
        raise NotImplementedError("Not implemented")

    return cell


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None,
                                               dtype=dtype)
        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        return tf.concat([output_fw, output_bw], axis=2)

# not used 
def _stack_encoder(cell_fw0, cell_fw1, cell_bw0, cell_bw1, inputs, sequence_length, dtype=None,  scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        # first forward laywer
        inputs_fw0 = inputs
        with tf.variable_scope("forward0"):
            output_fw0, state_fw0 = _gru_encoder(cell_fw0, inputs_fw0,
                                               sequence_length, None,
                                               dtype=dtype)
        # first backward laywer using first forward's output
        inputs_bw0 = tf.reverse_sequence(output_fw0, sequence_length,
                                        batch_axis=0, seq_axis=1) 
        
        with tf.variable_scope("backward0"):
            output_bw0, state_bw0 = _gru_encoder(cell_bw0, inputs_bw0,
                                               sequence_length, None,
                                               dtype=dtype)
        # final output by first backward
            output_bw0 = tf.reverse_sequence(output_bw0, sequence_length,
                                            batch_axis=0, seq_axis=1)
        # same as above
        inputs_bw1 = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)
        
        with tf.variable_scope("backward1"):
            output_bw1, state_bw1 = _gru_encoder(cell_bw1, inputs_bw1,
                                               sequence_length, None,
                                               dtype=dtype)
        inputs_fw1 = tf.reverse_sequence(output_bw1, sequence_length,
                                        batch_axis=0, seq_axis=1) 
        
        with tf.variable_scope("forward1"):
            output_fw1, state_fw1 = _gru_encoder(cell_fw1, inputs_fw1,
                                               sequence_length, None,
                                               dtype=dtype)

        return tf.concat([output_fw1, output_bw0], axis=2)
    

def _decoder(cell, inputs, memory, sequence_length, initial_state, dtype=None,
             scope=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        memory = tf.transpose(memory, [1, 0, 2])
        
        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        memory_ta = tf.TensorArray(tf.float32, tf.shape(memory)[0],
                                   tensor_array_name="memory_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
    
        input_ta = input_ta.unstack(inputs)
        memory_ta = memory_ta.unstack(memory)
        initial_state = layers.nn.linear(initial_state, output_size, True,
                                         False, scope="s_transform")
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, state):
            inp_t = input_ta.read(t)
            mem_t = memory_ta.read(t)

            cell_input = [inp_t, mem_t]
            cell_output, new_state = cell(cell_input, state)
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)

            out_ta = out_ta.write(t, cell_output)

            return t + 1, out_ta, new_state

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, initial_state)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])
        result = {
            "outputs": final_output,
            "initial_state": initial_state
        }

    return result


def model_graph(features, labels, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])
    char_vocab_size = len(params.vocabulary["char"])
    
    with tf.variable_scope("global_emb"):
        with tf.variable_scope("source_embedding"):
            if params.glove_emb_path:
                src_emb = tf.Variable(load_glove(params.glove_emb_path),
                                      name="embedding", 
                                      trainable=params.fine_tuning)
            else:
                src_emb = tf.get_variable("embedding",[src_vocab_size, params.embedding_size])
            src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])
            src_bias = tf.get_variable("bias", [params.embedding_size])
            src_inputs = tf.nn.bias_add(src_inputs, src_bias)
        
        with tf.variable_scope("target_embedding"):
            tgt_emb = tf.get_variable("embedding", [tgt_vocab_size, params.embedding_size])
            tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])
            tgt_bias = tf.get_variable("bias", [params.embedding_size])
            tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)
    
        with tf.variable_scope("char_embedding"):
            char_emb = tf.get_variable("embedding", [char_vocab_size, params.char_embedding_size])
            char_inputs = tf.nn.embedding_lookup(char_emb, features["chars"])
            mask_weights = tf.sequence_mask(features["char_length"])
            char_cnn_emb = layers.cnn.masked_conv1d_and_max(char_inputs, mask_weights, params.char_embedding_size, 3, params)
        
        src_inputs = tf.concat([src_inputs, char_cnn_emb], -1)
        src_inputs = layers.attention.add_timing_signal(src_inputs)
        tgt_inputs = layers.attention.add_timing_signal(tgt_inputs)
    
        if params.dropout and not params.use_variational_dropout:
            src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
            tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

        cell_fw = get_rnn_cell(params.rnn_cell, params.global_hidden_size, params.rnn_dropout, params.transition_num)
        cell_bw = get_rnn_cell(params.rnn_cell, params.global_hidden_size, params.rnn_dropout, params.transition_num)
        global_hidden = _encoder(cell_fw, cell_bw, src_inputs, features["source_length"])      
        # record the batch and sentence size
        batch_size = tf.shape(global_hidden)[0]
        sent_size = tf.shape(global_hidden)[1]

        if params.global_type == "mean":
            mask_weight = tf.expand_dims(features["source_length"], -1)
            mask_weight = tf.to_float(mask_weight)    
            global_emb = tf.reduce_sum(global_hidden, axis=-2) / mask_weight

        elif params.global_type == "max":
            float_mask = tf.to_float(features["source_length"])
            reshaped_mask = tf.reshape(float_mask, shape=[batch_size, 1, 1])
            tile_mask = tf.tile(reshaped_mask, [1, sent_size, params.hidden_size * 2])
            concat_hidden = tf.concat([global_hidden, tile_mask], axis=-2)
            global_emb = tf.map_fn(lambda x: tf.reduce_max(x[: tf.cast(x[-1][-1], tf.int32)], axis=-2), concat_hidden)

        elif params.global_type == "self":
            att_weight = layers.nn.linear(global_hidden, output_size=params.hidden_size, bias=True)
            float_mask = tf.to_float(tf.sequence_mask(features["source_length"]))
            append_inf_mask = (float_mask - 1) * 1e10
            expand_mask = tf.expand_dims(append_inf_mask, axis=-1)
            softmax_weight = tf.nn.softmax(expand_mask + att_weight, axis=1)
            global_emb = tf.reduce_sum(global_hidden * softmax_weight, axis=-2)

        # copy to each word position
        global_emb = tf.expand_dims(global_emb, axis=-2)
        global_emb = tf.tile(global_emb, [1, sent_size, 1])

    with tf.variable_scope("sequence_labeling"):
        with tf.variable_scope("source_embedding"):
            if params.glove_emb_path:
                src_emb = tf.Variable(load_glove(params.glove_emb_path),
                                      name="embedding",
                                      trainable=params.fine_tuning)
            else:
                src_emb = tf.get_variable("embedding", [src_vocab_size, params.embedding_size])
            src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])
            src_bias = tf.get_variable("bias", [params.embedding_size])
            src_inputs = tf.nn.bias_add(src_inputs, src_bias)

        with tf.variable_scope("bert_embedding"):
            if params.use_bert and params.bert_emb_path:
                src_bert_emb = features["bert"]  
                bert_bias = tf.get_variable("bert_bias", [params.bert_size])
                src_bert_emb = tf.convert_to_tensor(src_bert_emb, dtype=tf.float32)
                src_bert_emb = tf.nn.bias_add(src_bert_emb, bert_bias)
    
        with tf.variable_scope("target_embedding"):
            tgt_emb = tf.get_variable("embedding", [tgt_vocab_size, params.embedding_size])
            tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])
            tgt_bias = tf.get_variable("bias", [params.embedding_size])
            tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)
    
        with tf.variable_scope("char_embedding"):
            char_emb = tf.get_variable("embedding", [char_vocab_size, params.char_embedding_size])
            char_inputs = tf.nn.embedding_lookup(char_emb, features["chars"])
            mask_weights = tf.sequence_mask(features["char_length"])
            char_cnn_emb = layers.cnn.masked_conv1d_and_max(char_inputs, mask_weights, params.char_embedding_size, 3, params)

        concat_inputs = char_cnn_emb  # use char at least    
        if params.use_glove:
            concat_inputs = tf.concat([concat_inputs, src_inputs], -1)
        if params.use_bert:
            concat_inputs = tf.concat([concat_inputs, src_bert_emb], -1)

        src_inputs = tf.concat([concat_inputs, global_emb], -1)
        src_inputs = layers.attention.add_timing_signal(src_inputs)
        tgt_inputs = layers.attention.add_timing_signal(tgt_inputs)
    
        if params.dropout and not params.use_variational_dropout:
            src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
            tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)
        # encoder
        cell_fw = get_rnn_cell(params.rnn_cell, params.hidden_size, params.rnn_dropout, params.transition_num)
        cell_bw = get_rnn_cell(params.rnn_cell, params.hidden_size, params.rnn_dropout, params.transition_num)
        encoder_output = _encoder(cell_fw, cell_bw, src_inputs, features["source_length"])
    # decoder
    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
    src_mask = tf.sequence_mask(features["source_length"],
                                maxlen=tf.shape(encoder_output)[1],
                                dtype=tf.float32)
    src_mask = tf.expand_dims(src_mask, axis=2)
    initial_state = tf.reduce_sum(encoder_output * src_mask, axis=1) / tf.reduce_sum(src_mask, axis=1)

    # Shift left
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]

    cell = get_rnn_cell(params.rnn_cell, params.hidden_size, params.rnn_dropout, params.transition_num)
    decoder_output = _decoder(cell, shifted_tgt_inputs, encoder_output, length, initial_state)

    shifted_outputs = decoder_output["outputs"]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
    ]
    maxout_size = params.hidden_size // params.maxnum

    if labels is None:
        # Special case for non-incremental decoding
        maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
        ]
        readout = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                   concat=False)
        readout = tf.tanh(readout)

        if params.dropout and not params.use_variational_dropout:
            readout = tf.nn.dropout(readout, 1.0 - params.dropout)

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                                  scope="softmax")

        return logits

    readout = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                               concat=False)
    readout = tf.tanh(readout)

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction
    logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                              scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])

    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


class RNNsearch(interface.NMTModel):
    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearch, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"], params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.rnn_dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.rnn_dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            # model
            rnn_cell="DT",
            embedding_size=256,
            hidden_size=256,
            use_char_feature=False,
            char_embedding_size=128,
            maxnum=1,
            stack=False,
            bert_size=1024,
            use_bert=True,
            use_glove=True,
            fine_tuning=False,
            global_type="",
            global_hidden_size=256,
            global_transition_num=2,
            transition_num=4,
            # regularization
            dropout=0.5,
            rnn_dropout=0.3,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=False,
            batch_size=128,
            max_length=100,
            clip_grad_norm=5.0
        )

        return params

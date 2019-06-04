# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .nn import linear
from .nn import layer_norm

class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")

            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class LegacyGRUCell1(tf.nn.rnn_cell.RNNCell):
    """ change new_state = u * state + (1.0 - u) * tf.tanh(c)

    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell1, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")

            new_state = u * state + (1.0 - u) * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units



class StateToOutputWrapper(tf.nn.rnn_cell.RNNCell):
    """ Copy state to the output of RNNCell so that all states can be obtained
        when using tf.nn.dynamic_rnn

    :param cell: An instance of tf.nn.rnn_cell.RNNCell
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, cell, reuse=None):
        super(StateToOutputWrapper, self).__init__(_reuse=reuse)
        self._cell = cell

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)

        return (output, new_state), new_state

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return tuple([self._cell.output_size, self.state_size])


class AttentionWrapper(tf.nn.rnn_cell.RNNCell):
    """ Wrap an RNNCell with attention mechanism

    :param cell: An instance of tf.nn.rnn_cell.RNNCell
    :param memory: A tensor with shape [batch, mem_size, mem_dim]
    :param bias: A tensor with shape [batch, mem_size]
    :param attention_fn: A callable function with signature
        (inputs, state, memory, bias) -> (output, state, weight, value)
    :param output_weight: Whether to output attention weights
    :param output_value: Whether to output attention values
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, cell, memory, bias, attention_fn, output_weight=False,
                 output_value=False, reuse=None):
        super(AttentionWrapper, self).__init__(_reuse=reuse)
        memory.shape.assert_has_rank(3)
        self._cell = cell
        self._memory = memory
        self._bias = bias
        self._attention_fn = attention_fn
        self._output_weight = output_weight
        self._output_value = output_value

    def __call__(self, inputs, state, scope=None):
        outputs = self._attention_fn(inputs, state, self._memory, self._bias)
        cell_inputs, cell_state, weight, value = outputs
        cell_output, new_state = self._cell(cell_inputs, cell_state,
                                            scope=scope)

        if not self._output_weight and not self._output_value:
            return cell_output, new_state

        new_output = [cell_output]

        if self._output_weight:
            new_output.append(weights)

        if self._output_value:
            new_output.append(value)

        return tuple(new_output), new_state

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        if not self._output_weight and not self._output_value:
            return self._cell.output_size

        new_output_size = [self._cell.output_size]

        if self._output_weight:
            new_output_size.append(self._memory.shape[1])

        if self._output_value:
            new_output_size.append(self._memory.shape[2].value)

        return tuple(new_output_size)


class DL4MTGRULAUTransiLNCell(tf.nn.rnn_cell.RNNCell):
    """ DL4MT's implementation of GRUCell with LAU and Transition

    Args:
        num_units: int, The number of units in the RNN cell.
        reuse: (optional) Python boolean describing whether to reuse
            variables in an existing scope.  If not `True`, and the existing
            scope already has the given variables, an error is raised.
    """

    def __init__(self, num_transi, num_units, keep_prob=None, reuse=None):
        super(DL4MTGRULAUTransiLNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._keep_prob = keep_prob
        self._num_transi = num_transi

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(layer_norm(linear(all_inputs, self._num_units, False, False,
                                                scope="reset_gate"),
                                         scope="reset_gate_ln"))
            r2 = tf.nn.sigmoid(layer_norm(linear(all_inputs, self._num_units, False, False,
                                                 scope="reset_gate2"),
                                          scope="reset_gate2_ln"))
            u = tf.nn.sigmoid(layer_norm(linear(all_inputs, self._num_units, False, False,
                                                scope="update_gate"),
                                         scope="update_gate_ln"))
            linear_state = linear(state, self._num_units, True, False, scope="linear_state")
            linear_inputs = linear(inputs, self._num_units, False, False, scope="linear_inputs")
            linear_inputs_transform = linear(inputs, self._num_units, False, False, scope="linear_inputs_transform")
            c = tf.tanh(linear_inputs + r * linear_state) + r2 * linear_inputs_transform
            if self._keep_prob and self._keep_prob < 1.0:
                c = tf.nn.dropout(c, self._keep_prob)

            new_state = (1.0 - u) * state + u * c

        for i in range(int(self._num_transi)):
            rh = tf.nn.sigmoid(layer_norm(linear(new_state, self._num_units, False, False,
                                                 scope="trans_reset_gate_l%d" %i),
                                          scope="trans_reset_gate_ln_l%d" %i))
            uh = tf.nn.sigmoid(layer_norm(linear(new_state, self._num_units, False, False,
                                                 scope="trans_update_gate_l%d" %i),
                                          scope="trans_update_gate_ln_l%d" %i))
            ch = tf.tanh(rh * linear(new_state, self._num_units, True, False,
                                     scope="trans_candidate_l%d" %i))
            if self._keep_prob and self._keep_prob < 1.0:
                ch = tf.nn.dropout(ch, self._keep_prob)

            new_state = (1.0 - uh) * new_state + uh * ch

        return new_state, new_state


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import *
# from visualizer import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical#, accuracy
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed, Recurrent

def time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, units=None, timesteps=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        units: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not units:
        units = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b:
        x += b
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, units]))
        x.set_shape([None, None, units])
    else:
        x = K.reshape(x, (-1, timesteps, units))
    return x

class Attention(Recurrent):
    """Attention Recurrent Unit - Bengio et al. ICLR 2015.

    # Arguments
        units: dimension of the internal projections and the final output.
        h: we use it as attention to process the input
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    """
    def __init__(self, units, h, h_dim,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 #activation='tanh', inner_activation='hard_sigmoid',
                 #W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 #dropout_W=0., dropout_U=0., 
                 **kwargs):
        self.units = units
        self.h = h[:,-1,:]
        self.h_dim = h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        #self.activation = activations.get(activation)
        #self.inner_activation = activations.get(inner_activation)
        #self.W_regularizer = regularizers.get(W_regularizer)
        #self.U_regularizer = regularizers.get(U_regularizer)
        #self.b_regularizer = regularizers.get(b_regularizer)
        #self.dropout_W = dropout_W
        #self.dropout_U = dropout_U

        #if self.dropout_W or self.dropout_U:
        #    self.uses_learning_phase = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (units)
            self.states = [None]

        self.Wa = self.add_weight((self.units, self.units),
                                  initializer=self.kernel_initializer,
                                  name='{}_Wa'.format(self.name))
        self.Ua = self.add_weight((self.h_dim, self.units),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Ua'.format(self.name))
        self.Va = self.add_weight((self.units,1),
                                  initializer=self.kernel_initializer,
                                  name='{}_Va'.format(self.name))
        self.Wzr = self.add_weight((self.input_dim, 2 * self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Uzr = self.add_weight((self.units, 2 * self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Czr = self.add_weight((self.h_dim, 2 * self.units),
                                   initializer=self.recurrent_initializer,
                                   name='{}_Czr'.format(self.name))
        self.W = self.add_weight((self.input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name))
        self.U = self.add_weight((self.units, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_U'.format(self.name))
        self.C = self.add_weight((self.h_dim, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_C'.format(self.name))
        
        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):
        #if self.consume_less == 'cpu':
        #    input_shape = K.int_shape(x)
        #    input_dim = input_shape[2]
        #    timesteps = input_shape[1]

        #    x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    return K.concatenate([x_z, x_r, x_h], axis=2)
        #else:
        #    return x
        self.ha = time_distributed_dense(self.h, self.Ua)
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        #B_U = states[1]  # dropout matrices for recurrent units
        #B_W = states[2]
        h_tm1a = K.dot(h_tm1, self.Wa)
        eij = K.dot(K.tanh(K.repeat(h_tm1a, K.shape(self.h)[1]) + self.ha), self.Va)
        eijs = K.squeeze(eij, -1)
        alphaij = K.softmax(eijs) # batchsize * lenh       h batchsize * lenh * ndim
        ci = K.permute_dimensions(K.permute_dimensions(self.h, [2,0,1]) * alphaij, [1,2,0])
        cisum = K.sum(ci, axis=1)
        #print(K.shape(cisum), cisum.shape, ci.shape, self.h.shape, alphaij.shape, x.shape)

        zr = K.sigmoid(K.dot(inputs, self.Wzr) + K.dot(h_tm1, self.Uzr) + K.dot(cisum, self.Czr))
        zi = zr[:, :self.units]
        ri = zr[:, self.units: 2 * self.units]
        si_ = K.tanh(K.dot(inputs, self.W) + K.dot(ri*h_tm1, self.U) + K.dot(cisum, self.C))
        si = (1-zi) * h_tm1 + zi * si_
        return si, [si] #h_tm1, [h_tm1]
        '''if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            inner_z = matrix_inner[:, :self.units]
            inner_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.units:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.units:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.units]
                x_r = x[:, self.units: 2 * self.units]
                x_h = x[:, 2 * self.units:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]'''

    def get_constants(self, inputs, training=None):
        constants = []
        '''if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:'''
        constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(SimpleAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SimpleAttention(Recurrent):
    """Attention Recurrent Unit - Bengio et al. ICLR 2015.

    # Arguments
        units: dimension of the internal projections and the final output.
        h: we use it as attention to process the input
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    """
    def __init__(self, units, h, h_dim,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 #activation='tanh', inner_activation='hard_sigmoid',
                 #W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 #dropout_W=0., dropout_U=0., 
                 **kwargs):
        self.units = units
        self.h = h
        self.h_dim = h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        #self.activation = activations.get(activation)
        #self.inner_activation = activations.get(inner_activation)
        #self.W_regularizer = regularizers.get(W_regularizer)
        #self.U_regularizer = regularizers.get(U_regularizer)
        #self.b_regularizer = regularizers.get(b_regularizer)
        #self.dropout_W = dropout_W
        #self.dropout_U = dropout_U

        #if self.dropout_W or self.dropout_U:
        #    self.uses_learning_phase = True
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (units)
            self.states = [None]

        self.Wa = self.add_weight((self.units, self.units),
                                  initializer=self.kernel_initializer,
                                  name='{}_Wa'.format(self.name))
        self.Ua = self.add_weight((self.h_dim, self.units),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Ua'.format(self.name))
        self.Va = self.add_weight((self.units,1),
                                  initializer=self.kernel_initializer,
                                  name='{}_Va'.format(self.name))
        self.Wzr = self.add_weight((self.input_dim, 2 * self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Uzr = self.add_weight((self.units, 2 * self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Czr = self.add_weight((self.h_dim, 2 * self.units),
                                   initializer=self.recurrent_initializer,
                                   name='{}_Czr'.format(self.name))
        self.W = self.add_weight((self.input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name))
        self.U = self.add_weight((self.units, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_U'.format(self.name))
        self.C = self.add_weight((self.h_dim, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_C'.format(self.name))
        
        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):
        #if self.consume_less == 'cpu':
        #    input_shape = K.int_shape(x)
        #    input_dim = input_shape[2]
        #    timesteps = input_shape[1]

        #    x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    return K.concatenate([x_z, x_r, x_h], axis=2)
        #else:
        #    return x
        self.ha = K.dot(self.h, self.Ua) #time_distributed_dense(self.h, self.Ua)
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        #B_U = states[1]  # dropout matrices for recurrent units
        #B_W = states[2]
        h_tm1a = K.dot(h_tm1, self.Wa)
        eij = K.dot(K.tanh(h_tm1a + self.ha), self.Va)
        eijs = K.repeat_elements(eij, self.h_dim, axis=1)

        #alphaij = K.softmax(eijs) # batchsize * lenh       h batchsize * lenh * ndim
        #ci = K.permute_dimensions(K.permute_dimensions(self.h, [2,0,1]) * alphaij, [1,2,0])
        #cisum = K.sum(ci, axis=1)
        cisum = eijs*self.h
        #print(K.shape(cisum), cisum.shape, ci.shape, self.h.shape, alphaij.shape, x.shape)

        zr = K.sigmoid(K.dot(inputs, self.Wzr) + K.dot(h_tm1, self.Uzr) + K.dot(cisum, self.Czr))
        zi = zr[:, :self.units]
        ri = zr[:, self.units: 2 * self.units]
        si_ = K.tanh(K.dot(inputs, self.W) + K.dot(ri*h_tm1, self.U) + K.dot(cisum, self.C))
        si = (1-zi) * h_tm1 + zi * si_
        return si, [si] #h_tm1, [h_tm1]
        '''if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            inner_z = matrix_inner[:, :self.units]
            inner_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.units:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.units:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.units]
                x_r = x[:, self.units: 2 * self.units]
                x_h = x[:, 2 * self.units:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]'''

    def get_constants(self, inputs, training=None):
        constants = []
        '''if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:'''
        constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(SimpleAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SSimpleAttention(Recurrent):
    """Attention Recurrent Unit - Bengio et al. ICLR 2015.

    # Arguments
        units: dimension of the internal projections and the final output.
        h: we use it as attention to process the input
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    """
    def __init__(self, units, h, h_dim,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 #activation='tanh', inner_activation='hard_sigmoid',
                 #W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 #dropout_W=0., dropout_U=0., 
                 **kwargs):
        self.units = units
        self.h = h[:,-1,:]
        self.h_dim = h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        #self.activation = activations.get(activation)
        #self.inner_activation = activations.get(inner_activation)
        #self.W_regularizer = regularizers.get(W_regularizer)
        #self.U_regularizer = regularizers.get(U_regularizer)
        #self.b_regularizer = regularizers.get(b_regularizer)
        #self.dropout_W = dropout_W
        #self.dropout_U = dropout_U

        #if self.dropout_W or self.dropout_U:
        #    self.uses_learning_phase = True
        super(SSimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (units)
            self.states = [None]

        self.Wa = self.add_weight((self.units, self.units),
                                  initializer=self.kernel_initializer,
                                  name='{}_Wa'.format(self.name))
        self.Ua = self.add_weight((self.h_dim, self.units),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Ua'.format(self.name))
        self.Va = self.add_weight((self.units,1),
                                  initializer=self.kernel_initializer,
                                  name='{}_Va'.format(self.name))
        self.Wzr = self.add_weight((self.input_dim, 2 * self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Uzr = self.add_weight((self.units, 2 * self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Czr = self.add_weight((self.h_dim, 2 * self.units),
                                   initializer=self.recurrent_initializer,
                                   name='{}_Czr'.format(self.name))
        self.W = self.add_weight((self.input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name))
        self.U = self.add_weight((self.units, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_U'.format(self.name))
        self.C = self.add_weight((self.h_dim, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_C'.format(self.name))
        
        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):
        #if self.consume_less == 'cpu':
        #    input_shape = K.int_shape(x)
        #    input_dim = input_shape[2]
        #    timesteps = input_shape[1]

        #    x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
        #                                 input_dim, self.units, timesteps)
        #    return K.concatenate([x_z, x_r, x_h], axis=2)
        #else:
        #    return x
        self.ha = K.dot(self.h, self.Ua) #time_distributed_dense(self.h, self.Ua)
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        #B_U = states[1]  # dropout matrices for recurrent units
        #B_W = states[2]
        h_tm1a = K.dot(h_tm1, self.Wa)
        eij = K.dot(K.tanh(h_tm1a + self.ha), self.Va)
        eijs = K.repeat_elements(eij, self.h_dim, axis=1)

        #alphaij = K.softmax(eijs) # batchsize * lenh       h batchsize * lenh * ndim
        #ci = K.permute_dimensions(K.permute_dimensions(self.h, [2,0,1]) * alphaij, [1,2,0])
        #cisum = K.sum(ci, axis=1)
        cisum = eijs*self.h
        #print(K.shape(cisum), cisum.shape, ci.shape, self.h.shape, alphaij.shape, x.shape)

        zr = K.sigmoid(K.dot(inputs, self.Wzr) + K.dot(h_tm1, self.Uzr) + K.dot(cisum, self.Czr))
        zi = zr[:, :self.units]
        ri = zr[:, self.units: 2 * self.units]
        si_ = K.tanh(K.dot(inputs, self.W) + K.dot(ri*h_tm1, self.U) + K.dot(cisum, self.C))
        si = (1-zi) * h_tm1 + zi * si_
        return si, [si] #h_tm1, [h_tm1]
        '''if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            inner_z = matrix_inner[:, :self.units]
            inner_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.units:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.units:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.units]
                x_r = x[:, self.units: 2 * self.units]
                x_h = x[:, 2 * self.units:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]'''

    def get_constants(self, inputs, training=None):
        constants = []
        '''if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:'''
        constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(SSimpleAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SimpleAttention2(Recurrent):
    def __init__(self, units, h_dim,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 **kwargs):
        self.units = units
        self.h_dim = h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        super(SimpleAttention2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2] - self.h_dim

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None]

        self.Wa = self.add_weight((self.units, self.units),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Wa'.format(self.name))
        self.Ua = self.add_weight((self.h_dim, self.units),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Ua'.format(self.name))
        self.Va = self.add_weight((self.units,1),
                                  initializer=self.recurrent_initializer,
                                  name='{}_Va'.format(self.name))
        self.Wzr = self.add_weight((self.input_dim, 2 * self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Uzr = self.add_weight((self.units, 2 * self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_Wzr'.format(self.name))
        self.Czr = self.add_weight((self.h_dim, 2 * self.units),
                                   initializer=self.recurrent_initializer,
                                   name='{}_Czr'.format(self.name))
        self.W = self.add_weight((self.input_dim, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_W'.format(self.name))
        self.U = self.add_weight((self.units, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_U'.format(self.name))
        self.C = self.add_weight((self.h_dim, self.units),
                                 initializer=self.recurrent_initializer,
                                 name='{}_C'.format(self.name))
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):
        #self.ha = K.dot(self.h, self.Ua) #time_distributed_dense(self.h, self.Ua)
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        #B_U = states[1]  # dropout matrices for recurrent units
        #B_W = states[2]
        h_tm1a = K.dot(h_tm1, self.Wa)
        eij = K.dot(K.tanh(h_tm1a + K.dot(inputs[:, :self.h_dim], self.Ua)), self.Va)
        eijs = K.repeat_elements(eij, self.h_dim, axis=1)

        #alphaij = K.softmax(eijs) # batchsize * lenh       h batchsize * lenh * ndim
        #ci = K.permute_dimensions(K.permute_dimensions(self.h, [2,0,1]) * alphaij, [1,2,0])
        #cisum = K.sum(ci, axis=1)
        cisum = eijs*inputs[:, :self.h_dim]
        #print(K.shape(cisum), cisum.shape, ci.shape, self.h.shape, alphaij.shape, x.shape)

        zr = K.sigmoid(K.dot(inputs[:, self.h_dim:], self.Wzr) + K.dot(h_tm1, self.Uzr) + K.dot(cisum, self.Czr))
        zi = zr[:, :self.units]
        ri = zr[:, self.units: 2 * self.units]
        si_ = K.tanh(K.dot(inputs[:, self.h_dim:], self.W) + K.dot(ri*h_tm1, self.U) + K.dot(cisum, self.C))
        si = (1-zi) * h_tm1 + zi * si_
        return si, [si] #h_tm1, [h_tm1]

    def get_constants(self, inputs, training=None):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(SimpleAttention2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
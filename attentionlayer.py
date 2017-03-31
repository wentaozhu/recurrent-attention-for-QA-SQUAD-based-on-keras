import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Convolution3D, MaxPooling3D
from keras.layers import BatchNormalization, AveragePooling3D, UpSampling3D, Cropping3D, Merge
from keras.layers import SpatialDropout3D
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2, l1l2
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop, Adam
import numpy as np
from keras.layers.recurrent import Recurrent
from keras import initializations, activations, regularizers
from keras.engine import InputSpec

# copied from the keras github source. removed lots of unnecessary (for me) code

# assuming a 2D Convolution was run by hand before this layer.
# please note that this has no variables of its own.
# TODO: incorporate the 2D Convolution into this layer

class AttenLayer(Recurrent):
    def __init__(self, h, output_dim,
                 init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.h = h
        self.output_dim = output_dim
        #removing the regularizers and the dropout
        super(AttenLayer, self).__init__(**kwargs)
        # this seems necessary in order to accept 3 input dimensions
        # (samples, timesteps, features)
        self.input_spec=[InputSpec(ndim=3)]
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        # moved here from the constructor. Curiously it seems like batch_size has been removed from here.
        self.output_dim=[1,input_shape[1], self.output_dim
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        self.Wa = self.add_weight(shape=(input_shape[2], input_shape[2]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Ua = self.add_weight(shape=(input_shape[2], input_shape[2]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Va = self.add_weight(shape=(input_shape[2],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.W = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.U = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.C = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wz = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Uz = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Cz = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wr = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Ur = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Cr = self.add_weight(shape=(input_shape[2], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
    def reset_states(self):
        # TODO: the state must become 2D. am I doing this right ?
        # TODO: assuming that the first dimension is batch_size, I'm now hardcoding for 2D images and th layout
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.input_shape[2], self.input_shape[2])))
        else:
            self.states = [K.zeros((input_shape[0], self.input_shape[2], self.input_shape[2]))]

    def preprocess_input(self, x):
        # here was a distinction between cpu and gpu. for starters I'll just use the cpu code
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        #TODO: I've got no idea where the states are set. Maybe in the superclass ?
        #TODO: debug to see how many entries there really are in the states variable
        #B_U = states[1]  # dropout matrices for recurrent units
        #B_W = states[2]

        #TODO: now I need to use the features from the Convolution.#
        # Since I'll hardcode for th layout, the x will (hopefully) look like:
        # [batch, features, x_dim, y_dim]
        # note: slicing is possible, just always use : the entire first dimension (batch)
        """
        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        """
        #all the code above produces z, r, and hh.
        # I would like to use the values produced by the convolution instead
        # just drop all of the code above and slice the input

        #TODO: add the activations here
        z=self.inner_activation(x[:,0,:,:])
        r=self.inner_activation(x[:,1,:,:])
        hh=self.activation(x[:,2,:,:])

        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        #dropping all of this. There us no dropout or anything else in this layer.
        #TODO: do I need to have this method at all. It overrides something from super.
        #might be better to stick with the inherited method if I don't do anything here.
        return constants

    def get_initial_states(self, x):
        initial_state=K.zeros_like(x)   # (samples, timesteps, input_dim)
                                        # input_dim = (3, x_dim, y_dim)
        initial_state=K.sum(initial_state, axis=(1,2)) # (samples, x_dim, y_dim)
        return initial_state


    def get_output_shape_for(self, input_shape):
        #TODO: this is hardcoding for th layout
        return (input_shape[0],1,input_shape[2],input_shape[3])

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__}

        # removed the various regularizers and dropouts.
        # surely this isn't needed if not present ?
        base_config = super(CGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
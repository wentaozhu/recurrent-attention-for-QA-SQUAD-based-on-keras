from utils import JsonUtil
from pprint import pprint
import numpy as np
import re

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector, TimeDistributed
from keras.layers import recurrent, Input, Bidirectional, LSTM
from keras.layers.pooling import AveragePooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from layers import Attention, AttentionDecoderCell
from recurrentshop import LSTMCell, RecurrentContainer
def loadGloveModel(gloveFile):
    print "Loading Glove Model..."
    f = open(gloveFile,'r')
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs 
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def splitDatasets(f):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices, 
       and keep track of max context and question lengths.
    '''
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xAnswerBegin = [] # list of indices of the beginning word in each answer span
    xAnswerEnd = [] # list of indices of the ending word in each answer span
    xAnswerText = [] # list of the answer text
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f:
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            contextTokenized = tokenize(context)
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                questionTokenized = tokenize(question)
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa._id_
                answers = qa._answers_
                for answer in answers:
                    answerText = answer._text_
                    answerTokenized = tokenize(answerText)
                    # find indices of beginning/ending words of answer span among tokenized context
                    contextToAnswerFirstWord = context[:answer._answer_start_ + len(answerTokenized[0])]
                    answerBeginIndex = len(tokenize(contextToAnswerFirstWord)) - 1
                    answerEndIndex = answerBeginIndex + len(answerTokenized) - 1
                    
                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)
    return xContext, xQuestion, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion


def vectorizeData(xContext, xQuestion, xAnswerBeing, xAnswerEnd, word_index, context_maxlen, question_maxlen):
    '''Vectorize the words to their respective index and pad context to max context length and question to max question length.
       Answers vectors are padded to the max context length as well.
    '''
    X = []
    Xq = []
    YBegin = []
    YEnd = []
    for i in xrange(len(xContext)):
        x = [word_index[w] for w in xContext[i]]
        xq = [word_index[w] for w in xQuestion[i]]
        # map the first and last words of answer span to one-hot representations
        y_Begin =  np.zeros(len(xContext[i]))
        y_Begin[xAnswerBeing[i]] = 1
        y_End = np.zeros(len(xContext[i]))
        y_End[xAnswerEnd[i]] = 1
        X.append(x)
        Xq.append(xq)
        YBegin.append(y_Begin)
        YEnd.append(y_End)
    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), pad_sequences(YBegin, maxlen=context_maxlen, padding='post'), pad_sequences(YEnd, maxlen=context_maxlen, padding='post')

# Note: Need to download and unzip Glove pre-train model files into same file as this script
GloveDimOption = '50' # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
embeddings_index = loadGloveModel('../data/glove.6B/glove.6B.' + GloveDimOption + 'd.txt')  

# load training data, parse, and split
print('Loading in training data...')
#trainData = JsonUtil.import_qas_data('../data/test.json')
trainData = JsonUtil.import_qas_data('../data/train-v1.1.json')
tContext, tQuestion, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)
print(maxLenTContext, maxLenTQuestion)

# load validation data, parse, and split
print('Loading in Validation data...')
#valData = JsonUtil.import_qas_data('../data/test.json')
valData = JsonUtil.import_qas_data('../data/dev-v1.1.json')
vContext, vQuestion, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = splitDatasets(valData)
print(maxLenTContext, maxLenVQuestion)

print('Building vocabulary...')
# build a vocabular over all training and validation context paragraphs and question words
vocab = {}
for words in tContext + tQuestion + vContext + vQuestion:
    for word in words:
        if word not in vocab:
            vocab[word] = 1
vocab = sorted(vocab.keys())  
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, maxLenVContext)
question_maxlen = max(maxLenTQuestion, maxLenVQuestion)

# vectorize training and validation datasets
print('Begin vectoring process...')
#tX: training Context, tXq: training Question, tYBegin: training Answer Begin ptr, tYEnd: training Answer End ptr
tX, tXq, tYBegin, tYEnd = vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)
#vX: validation Context, vXq: validation Question, vYBegin: validation Answer Begin ptr, vYEnd: validation Answer End ptr
vX, vXq, vYBegin, vYEnd = vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index, context_maxlen, question_maxlen)
print('Vectoring process completed.')

print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tYBegin.shape = {}'.format(tYBegin.shape))
print('tYEnd.shape = {}'.format(tYEnd.shape))
print('vX.shape = {}'.format(vX.shape))
print('vXq.shape = {}'.format(vXq.shape))
print('vYBegin.shape = {}'.format(vYBegin.shape))
print('vYEnd.shape = {}'.format(vYEnd.shape))
print('context_maxlen, question_maxlen = {}, {}'.format(context_maxlen, question_maxlen))

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
EMBEDDING_DIM = int(GloveDimOption)
MAX_SEQUENCE_LENGTH = context_maxlen

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
print('Embedding matrix completed.')

# -------------- DNN goes after here ---------------------
cinput = Input(shape=(context_maxlen,), dtype='int32')
cembed = embedding_layer(cinput)
clstm1 = Bidirectional(LSTM(100, return_sequences=True))(cembed)

qinput = Input(shape=(question_maxlen,), dtype='int32')
qembed = embedding_layer(qinput)
qlstm1 = Bidirectional(LSTM(100, return_sequences=True))(qembed)

cdecoder = RecurrentContainer(decode=True, output_length=context_maxlen, input_length=context_maxlen)
cdecoder.add(AttentionDecoderCell(output_dim=100, hidden_dim=100))
clstm2 = cdecoder(clstm1)
ch1 = Attention(qlstm1)(clstm1)
clstm2 = Bidirectional(LSTM(100, return_sequences=True))(ch1)

qh1 = Attention(clstm2)(qlstm1)
qlstm2 = Bidirectional(LSTM(100, return_sequences=True))(qh1)

ch2 = Attention(qlstm2)(clstm2)
qh2 = Attention(ch2)(qlstm2)

h = Merge([ch2, qh2], mode='concat')
hlstm = Bidirectional(LSTM(100))(h)
output1 = Dense(context_maxlen, activation='softmax')(hlstm)
hmerge = Merge([hlstm, output1], mode='concat')
output2 = Dense(context_maxlen, activation='softmax')(hmerge)

qnamodel = Model(input=[cinput, qinput], output=[output1, output2])
adam = Adam(lr=0.0003)
qnamodel.compile(optimizer=adam,
              loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
              loss_weights={'output1': 1., 'output2': 1.},
              metrics=['weighted_accuracy'])
qnamodel.summary()
best_model_file = './squad3e-4.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_output1_weighted_accuracy', verbose=1, save_best_only = True)
# and trained it via:
print(tX.shape, tXq.shape, tYBegin.shape, tYEnd.shape, vX.shape, vXq.shape,
     vYBegin.shape, vYEnd.shape)
qnamodel.fit({'cinput': tX, 'qinput': tXq},
          {'output1': tYBegin, 'output2': tYEnd},nb_epoch=100, batch_size=128,
          validation_data=({'cinput': vX, 'qinput': vXq},
                           {'output1': vYBegin, 'output2': vYEnd}), callbacks=[best_model], verbose=2)

from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param

        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.inner_init((self.output_dim, 1),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants


class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.layer.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.layer.inner_init((attention_dim, self.layer.output_dim), name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.layer.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.layer.inner_init((self.layer.output_dim, 1), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights = [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
return last_output

class GRU(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(GRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':
            self.W = self.add_weight((self.input_dim, 3 * self.output_dim),
                                     initializer=self.init,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer)
            self.U = self.add_weight((self.output_dim, 3 * self.output_dim),
                                     initializer=self.inner_init,
                                     name='{}_U'.format(self.name),
                                     regularizer=self.U_regularizer)
            self.b = self.add_weight((self.output_dim * 3,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)
        else:
            self.W_z = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_z'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_z = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_z'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_z = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_z'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_r = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_r'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_r = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_r'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_r = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_r'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_h = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_h'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_h = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_h'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_h = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_h'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
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
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

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
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
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
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(GRU, self).get_config()
return dict(list(base_config.items()) + list(config.items()))
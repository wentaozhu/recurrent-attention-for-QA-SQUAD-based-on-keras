from utils import JsonUtil
from pprint import pprint
import numpy as np
import re

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector, merge
from keras.layers import recurrent, Input, Bidirectional, LSTM, Lambda
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from rnnlayer import Attention
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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

# load validation data, parse, and split
print('Loading in Validation data...')
#valData = JsonUtil.import_qas_data('../data/test.json')
valData = JsonUtil.import_qas_data('../data/dev-v1.1.json')
vContext, vQuestion, vAnswerBegin, vAnswerEnd, vAnswerText, maxLenVContext, maxLenVQuestion = splitDatasets(valData)

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
                            trainable=True, name='embedding')
print('Embedding matrix completed.')

# -------------- DNN goes after here ---------------------

cinput = Input(shape=(context_maxlen,), dtype='int32', name='cinput')
cembed = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],
                   input_length=maxLenTContext,
                   trainable=True, name='cembedding')(cinput)

qinput = Input(shape=(question_maxlen,), dtype='int32', name='qinput')
qembed = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],
                   input_length=maxLenTQuestion,
                   trainable=True, name='qembedding')(qinput)
qlstm1 = Bidirectional(LSTM(100, return_sequences=True, name='qlstm1'))(qembed)
#qlstm1 = Bidirectional(LSTM(100, return_sequences=True))(qembed)
print(qlstm1.shape)
clstm1 = Attention(100, qlstm1, 200, return_sequences=True, name='clstm1')(cembed)
qlstm2 = Attention(100, clstm1, 100, return_sequences=True, name='qlstm2')(qlstm1)
clstm2 = Attention(100, qlstm2, 100, return_sequences=True, name='clstm2')(clstm1)
qlstm3 = Attention(100, clstm2, 100, return_sequences=True, name='qlstm3')(qlstm2)
clstm3 = Attention(100, qlstm3, 100, return_sequences=False, name='clstm3')(clstm2)
qlstm4 = Bidirectional(LSTM(100, name='qlstm4'))(qlstm3)
h = merge([qlstm4, clstm3], mode='concat', name='merge1')
#qlstm3last = Lambda(lambda x: K.squeeze(x[:,-1,:], 1), output_shape=lambda s: (s[0], s[2]))(qlstm3)
#h = Lambda(lambda x, y: K.concatenate([x,y], axis=-1), 
#           output_shape=lambda s: (s[0], s[1]*2))(qlstm3last, clstm3)
#clstm3last = Lambda(lambda x: K.squeeze(x[:,-1,:], 1), output_shape=lambda s: (s[0], s[2]))(clstm3)
#h = merge([K.squeeze(qlstm3[:,-1,:], 1), clstm3], mode='concat', name='merge1')
#merged = Merge([clstm3, qlstm3], mode=lambda x: x[0] - x[1])
#clstm1 = Bidirectional(Attention(100, qlstm1, 200, return_sequences=True))(cembed)

#qlstm2 = Bidirectional(Attention(100, clstm1, 200, return_sequences=True))(qlstm1)

#clstm2 = Bidirectional(Attention(100, qlstm2, 200, return_sequences=True))(clstm1)

#h = merge([clstm2, K.repeat(K.mean(qlstm2, axis=1), n=maxLenTContext)], mode='concat', name='merge1')
#qlstm2reshap = K.reshape(K.repeat(K.mean(qlstm2, axis=1), n=maxLenTContext), (-1, 100))
#clstm2reshap = K.reshape(clstm2, (-1, 100))
#hreshap = merge([clstm2reshap, qlstm2reshap], mode='concat', name='merge1')
#h = K.reshape(hreshap, (-1, maxLenTContext, 200))
#hlstm = Bidirectional(LSTM(100, return_sequences=True, name='hlstm'))(h)
#hlstm1 = LSTM(100, name='hlstm1')(h)
output1 = Dense(context_maxlen, activation='softmax', name='output1')(h)
hmerge = merge([h, output1], mode='concat', name='merge2')
output2 = Dense(context_maxlen, activation='softmax', name='output2')(hmerge)

qnamodel = Model(input=[cinput, qinput], output=[output1, output2])
adam = Adam(lr=0.0003)
qnamodel.compile(optimizer=adam,
              loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
              loss_weights={'output1': 1., 'output2': 1.},
              metrics=['accuracy'])
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
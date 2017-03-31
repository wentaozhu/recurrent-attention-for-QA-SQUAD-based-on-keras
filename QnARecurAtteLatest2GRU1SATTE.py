from utils import JsonUtil
import json
from pprint import pprint
import numpy as np
import re
import io
import nltk

from keras import backend as K
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Merge, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import probas_to_classes
import theano
from rnnlayer import Attention, SimpleAttention, SSimpleAttention
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

''' written by wentao, revised from scott'''
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
    return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]

def tokenizeVal(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    tokenizedSent = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]
    tokenIdx2CharIdx = [None] * len(tokenizedSent)
    idx = 0
    token_idx = 0
    while idx < len(sent) and token_idx < len(tokenizedSent):
        word = tokenizedSent[token_idx]
        if sent[idx:idx+len(word)] == word:
            tokenIdx2CharIdx[token_idx] = idx
            idx += len(word)
            token_idx += 1 
        else:
            idx += 1
    return tokenizedSent, tokenIdx2CharIdx


def splitDatasets(f):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices, 
       and keep track of max context and question lengths.
    '''
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xQuestion_id = [] # list of question id
    xAnswerBegin = [] # list of indices of the beginning word in each answer span
    xAnswerEnd = [] # list of indices of the ending word in each answer span
    xAnswerText = [] # list of the answer text
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f:
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized = tokenize(context.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa._id_
                answers = qa._answers_
                for answer in answers:
                    answerText = answer._text_
                    answerTokenized = tokenize(answerText.lower())
                    # find indices of beginning/ending words of answer span among tokenized context
                    contextToAnswerFirstWord = context1[:answer._answer_start_ + len(answerTokenized[0])]
                    answerBeginIndex = len(tokenize(contextToAnswerFirstWord.lower())) - 1
                    answerEndIndex = answerBeginIndex + len(answerTokenized) - 1
                    
                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xQuestion_id.append(str(question_id))
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)
    return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion

# for validation dataset, as there's no need to keep track of answers
def splitValDatasets(f):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices, 
       and keep track of max context and question lengths.
    '''
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xQuestion_id = [] # list of question id
    xToken2CharIdx = []
    xContextOriginal = []
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f:
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized, tokenIdx2CharIdx = tokenizeVal(context1.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa._id_
                answers = qa._answers_
                
                xToken2CharIdx.append(tokenIdx2CharIdx)
                xContextOriginal.append(context)
                xContext.append(contextTokenized)
                xQuestion.append(questionTokenized)
                xQuestion_id.append(str(question_id))

    return xContext, xToken2CharIdx, xContextOriginal, xQuestion, xQuestion_id, maxLenContext, maxLenQuestion


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

# for validation dataset
def vectorizeValData(xContext, xQuestion, word_index, context_maxlen, question_maxlen):
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

        X.append(x)
        Xq.append(xq)

    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post')

def transpose(x):
    return K.permute_dimensions(x, (0,2,1))

def transpose_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0], shape[2], shape[1])


def bd(inputs):
    x,y = inputs
    result = K.batch_dot(x,y,axes=[1,1])
    return result

def bd_output_shape(input_shape):
    shape = list(input_shape)
    #print('$$$$$$$$$$$$$$$$$$')
    #print(shape)
    #print(shape[0][0], shape[0][2], shape[1][2])
    return (shape[0][0], shape[0][2], shape[1][2])
    
    
def bd2(inputs):
    x,y = inputs
    result = K.batch_dot(x,y,axes=[2,1])
    return result

def bd2_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0][0], shape[0][1], shape[1][2])

def bd3(inputs):
    x,y = inputs
    result = K.batch_dot(x,y,axes=[1,2])
    return result

def bd3_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0][0], shape[0][2], shape[1][1])

def bd4(inputs):
    x,y = inputs
    result = K.batch_dot(x,y,axes=[2,2])
    return result

def bd4_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0][0], shape[0][1], shape[1][1])        
# Note: Need to download and unzip Glove pre-train model files into same file as this script
GloveDimOption = '50' # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
embeddings_index = loadGloveModel('../data/glove.6B.' + GloveDimOption + 'd.txt')  

# load training data, parse, and split
print('Loading in training data...')
#trainData = JsonUtil.import_qas_data('../data/test2.json')
trainData = JsonUtil.import_qas_data('../data/train-v1.1.json')
tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)

# load validation data, parse, and split
print('Loading in Validation data...')
#valData = JsonUtil.import_qas_data('../data/test2.json')
valData = JsonUtil.import_qas_data('../data/dev-v1.1.json')
vContext, vToken2CharIdx, vContextOriginal, vQuestion, vQuestion_id, maxLenVContext, maxLenVQuestion = splitValDatasets(valData)

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
# shuffle train data
randindex = np.random.permutation(tX.shape[0])
tX = tX[randindex, :]
tXq = tXq[randindex, :]
tYBegin = tYBegin[randindex, :]
tYEnd = tYEnd[randindex, :]
#vX: validation Context, vXq: validation Question
vX, vXq = vectorizeValData(vContext, vQuestion, word_index, context_maxlen, question_maxlen)
print('Vectoring process completed.')

print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tYBegin.shape = {}'.format(tYBegin.shape))
print('tYEnd.shape = {}'.format(tYEnd.shape))
print('vX.shape = {}'.format(vX.shape))
print('vXq.shape = {}'.format(vXq.shape))
print('context_maxlen, question_maxlen = {}, {}'.format(context_maxlen, question_maxlen))

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
EMBEDDING_DIM = int(GloveDimOption)
MAX_SEQUENCE_LENGTH = context_maxlen

embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
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

''' Question and Answer Encoder '''
question_input = Input(shape=(question_maxlen,), dtype='int32', name='question_input')
#question_input = Input(batch_shape=(None,), dtype='float32', name='question_input')
context_input = Input(shape=(context_maxlen,), dtype='int32', name='context_input')
#context_input = Input(batch_shape=(None,), dtype='float32', name='context_input')

questionEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                         mask_zero=True, weights=[embedding_matrix], 
                         input_length=question_maxlen, trainable=False)(question_input)
contextEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                         mask_zero=True, weights=[embedding_matrix], 
                         input_length=context_maxlen, trainable=False)(context_input)

Q = Bidirectional(GRU(64, return_sequences=True))(questionEmbd)
D = Bidirectional(GRU(64, return_sequences=True))(contextEmbd)
Q1 = Bidirectional(GRU(96, return_sequences=True))(Q)
D1 = Bidirectional(GRU(96, return_sequences=True))(D)
Q2 = GRU(128, return_sequences=False)(Q1)
D2 = SimpleAttention(128, Q2, 128, return_sequences=False)(D1)
L = merge([D2, Q2], mode='concat')

answerPtrBegin_output = Dense(context_maxlen, activation='softmax')(L)
Lmerge = merge([L, answerPtrBegin_output], mode='concat', name='merge2')
answerPtrEnd_output = Dense(context_maxlen, activation='softmax')(Lmerge)


model = Model(input=[context_input, question_input], output=[answerPtrBegin_output, answerPtrEnd_output])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[.04, 0.04], metrics=['accuracy'])
model.summary()
# checkpoint
#filepath1="1GRU1satte128dense1weights-improvement-{epoch:02d}-{val_dense_1_acc:.2f}.hdf5"
#checkpoint1 = ModelCheckpoint(filepath1, monitor='val_dense_1_acc', verbose=1, save_best_only=True, mode='max')
#filepath2="1GRU1satte128dense2weights-improvement-{epoch:02d}-{val_dense_2_acc:.2f}.hdf5"
#checkpoint2 = ModelCheckpoint(filepath2, monitor='val_dense_2_acc', verbose=1, save_best_only=True, mode='max')
checkpoint3 = EarlyStopping(monitor='val_dense_1_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')
checkpoint4 = EarlyStopping(monitor='val_dense_2_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')
callbacks_list = [checkpoint3, checkpoint4] #, checkpoint3, checkpoint4]
model.fit([tX, tXq], [tYBegin, tYEnd], epochs=100, batch_size=128, shuffle=True, validation_split=0.2, \
          callbacks=callbacks_list)
predictions = model.predict([vX, vXq], batch_size=128)
#------------------- DNN Ends here -----------------------------------------
print(predictions[0].shape, predictions[1].shape)
# make class prediction
ansBegin = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ansEnd = np.zeros((predictions[0].shape[0],),dtype=np.int32) 
for i in xrange(predictions[0].shape[0]):
	ansBegin[i] = predictions[0][i, :].argmax()
	ansEnd[i] = predictions[1][i, :].argmax()
print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())
# extract answer tokens and join them
answers = {}
for i in xrange(len(vQuestion_id)):
    #print i
    if ansBegin[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = ""
    elif ansEnd[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:]
    else:
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:vToken2CharIdx[i][ansEnd[i]]+len(vContext[i][ansEnd[i]])]

# write out answers to json file
with io.open('../data/2biGRU1satte128dev-prediction.json', 'w', encoding='utf-8') as f:
    f.write(unicode(json.dumps(answers, ensure_ascii=False)))

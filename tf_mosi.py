import data_loader as loader
import numpy as np
seed = 123
np.random.seed(seed)

from theano import function, shared, pp
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import defaultdict, OrderedDict
import argparse
import cPickle as pickle

from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from theano.tensor.shared_randomstreams import RandomStreams
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras import optimizers
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from sklearn.metrics import mean_absolute_error
from keras.optimizers import SGD

import sys
sys.path.append('../../model/')
from att_lstm_lg import train_att_lstm_reg#, train_att_lstm_acc


parser = argparse.ArgumentParser()
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--l2', default=0.0, type=float)
parser.add_argument('--lr', default=0.01, type=float)		# 0.0005
parser.add_argument('--units', default=160, type=int)		# 160
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--loss', default='mae', type=str)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--optimizer', default='adam', type=str)  #sgd or adam
parser.add_argument('--toLabel', default='-', type=str)  # - or 5
parser.add_argument('--config', default='configs/test.json', type=str)
parser.add_argument('--train_patience', default=100, type=int)
parser.add_argument('--pretrain', default=0, type=int, choices=[0, 1], help='0: use all the modalities; 1: use only text, other modalities are set as 0')
parser.add_argument('-f', '--feature', default=[], type=list, help='what features to use besides text. c: covarep; f: facet. default is null')
parser.add_argument('-a', '--attention', default=0, type=int, choices=[0,1], help='whether to use attention model. 1: use; 0: not. default is 0')
parser.add_argument('-s', '--feature_selection', default=1, type=int, choices=[0,1], help='whether to use feature_selection')
parser.add_argument('-c', '--convolution', default=0, type=int, choices=[0,1], help='whether to use convolutional layer on covarep and facet')

args = parser.parse_args()


val_split = 0.1514                      # fixed. 52 training 10 validation
tr_split = 2.0/3
train_epoch = 1000

weights_folder_path = 'weights/'
train_patience = 5
max_segment_len = 20 #The max length of a segment in dataset is 114 
min_word_frequency = 1
embedding_vecor_length = 300
use_pretrained_word_embedding = True
use_pretrained_single_model = True
use_cartesian_fusion = True
end_to_end = True
feature_selection = True
a, b=4.0, 8.0

fusion_method = 'Cartesion' if use_cartesian_fusion else 'Concat'

#word2ix = loader.load_word2ix()
word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
train, valid, test = loader.load_word_level_features(max_segment_len, tr_split)

feature_str = ''
if args.feature_selection:
	with open('/media/bighdd5/Paul/mosi/fs_mask.pkl') as f:
		[covarep_ix, facet_ix] = pickle.load(f)
	facet_train = train['facet'][:,:,facet_ix]
	facet_valid = valid['facet'][:,:,facet_ix]
	facet_test = test['facet'][:,:,facet_ix]
	covarep_train = train['covarep'][:,:,covarep_ix]
	covarep_valid = valid['covarep'][:,:,covarep_ix]
	covarep_test = test['covarep'][:,:,covarep_ix]
	feature_str = '_t'+str(embedding_vecor_length) + '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
else:
	facet_train = train['facet']
	facet_valid = valid['facet']
	covarep_train = train['covarep'][:,:,1:35]
	covarep_valid = valid['covarep'][:,:,1:35]
	facet_test = test['facet']
	covarep_test = test['covarep'][:,:,1:35]
text_train = train['text']
text_valid = valid['text']
text_test = test['text']
y_train = train['label']
y_valid = valid['label']
y_test = test['label']


facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
facet_train_max[facet_train_max==0] = 1
#covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
#covarep_train_max[covarep_train_max==0] = 1

facet_train = facet_train / facet_train_max
facet_valid = facet_valid / facet_train_max
#covarep_train = covarep_train / covarep_train_max
facet_test = facet_test / facet_train_max
#covarep_test = covarep_test / covarep_train_max


weights_folder_path = 'weights/'
weights_path = weights_folder_path + ''.join(sorted(args.feature)) + ("_attention" if args.attention else "") + ("_conv" if args.convolution else "") + feature_str +".h5"


if args.pretrain:
	# use only text to pretrain the model
	facet_test, facet_valid, facet_train = np.zeros(facet_test.shape), np.zeros(facet_valid.shape), np.zeros(facet_train.shape)
	covarep_test, covarep_valid, covarep_train = np.zeros(covarep_test.shape), np.zeros(covarep_valid.shape), np.zeros(covarep_train.shape)

print text_train.shape
covarep_train = np.mean(covarep_train, axis=1)
print covarep_train.shape
facet_train = np.mean(facet_train, axis=1)
print facet_train.shape
X_train = np.concatenate((covarep_train, facet_train, text_train), axis=1)

f_Facet_num = facet_train.shape[1]
f_Covarep_num = covarep_train.shape[1]

#print text_valid.shape
covarep_valid = np.mean(covarep_valid, axis=1)
#print covarep_valid.shape
facet_valid = np.mean(facet_valid, axis=1)
#print facet_valid.shape
X_valid = np.concatenate((covarep_valid, facet_valid, text_valid), axis=1)

X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)


print text_test.shape
covarep_test = np.mean(covarep_test, axis=1)
print covarep_test.shape
facet_test = np.mean(facet_test, axis=1)
print facet_test.shape
X_test = np.concatenate((covarep_test, facet_test, text_test), axis=1)


X_Covarep_test = X_test[:, :f_Covarep_num]
X_Facet_test = X_test[:, f_Covarep_num: f_Facet_num + f_Covarep_num]
X_text_test = X_test[:, f_Facet_num + f_Covarep_num:]
X_Covarep_train = X_train[:, :f_Covarep_num]
X_Facet_train = X_train[:, f_Covarep_num: f_Facet_num + f_Covarep_num]
X_text_train = X_train[:, f_Facet_num + f_Covarep_num:]

Covarep_model = Sequential()
Covarep_model.add(BatchNormalization(input_shape=(f_Covarep_num,), name = 'covarep_layer_0'))
Covarep_model.add(Dropout(0.2, name = 'covarep_layer_1'))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_2', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_3', trainable=end_to_end))
Covarep_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'covarep_layer_4', trainable=end_to_end))
#Covarep_model.add(Dense(1, name = 'covarep_layer_5'))

Facet_model = Sequential()
Facet_model.add(BatchNormalization(input_shape=(f_Facet_num,), name = 'facet_layer_0'))
Facet_model.add(Dropout(0.2, name = 'facet_layer_1'))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_2', trainable=end_to_end))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_3', trainable=end_to_end))
Facet_model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), name = 'facet_layer_4', trainable=end_to_end))
#Facet_model.add(Dense(1, name = 'facet_layer_5'))

text_model = Sequential()
text_model.add(Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_layer_0', trainable=end_to_end))
text_model.add(LSTM(128, name = 'text_layer_1', trainable=end_to_end))
text_model.add(Dense(64, name = 'text_layer_2', W_regularizer=l2(0.0), trainable=end_to_end))


if use_cartesian_fusion:
	bias_model=Sequential()
	bias_model.add(Reshape((1,),input_shape=(1,)))
	biased_Covarep = Merge([bias_model, Covarep_model], mode='concat')
	biased_Facet = Merge([bias_model, Facet_model], mode='concat')
	biased_text= Merge([bias_model, text_model], mode='concat')
	Covarep_biased_model, Facet_biased_model, text_biased_model = Sequential(), Sequential(), Sequential()
		
	Covarep_biased_model.add(biased_Covarep)
	Covarep_biased_model.add(Reshape((1, 32 + 1)))
	Facet_biased_model.add(biased_Facet)
	Facet_biased_model.add(Reshape((1, 32 + 1)))
	text_biased_model.add(biased_text)
	text_biased_model.add(Reshape((1, 64 + 1)))
	dot_layer1 = Merge([Covarep_biased_model, Facet_biased_model], mode='dot', dot_axes=1, name='dot_layer_1') 
	dot_layer1_reshape = Reshape((1, (32 + 1) * (32 + 1)), name='5')
	fusion_model_tmp = Sequential()
	fusion_model_tmp.add(dot_layer1)
	fusion_model_tmp.add(dot_layer1_reshape)
	dot_layer2=Merge([fusion_model_tmp,text_biased_model], mode='dot', dot_axes=1, name='dot_layer_2')
	fusion_model = Sequential()
	fusion_model.add(dot_layer2)
	fusion_model.add(Reshape(((32 + 1) * (32 + 1) * (64 + 1),)))
	fusion_model.add(Dropout(args.dropout))
	fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(args.l2), name = 'fusion_layer_1'))
	fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(args.l2), name = 'fusion_layer_2'))
	#fusion_model.add(Dense(128, activation='relu', W_regularizer=l2(0.01),name = 'fusion_layer_3'))
	fusion_model.add(Dense(1, activation='sigmoid',W_regularizer=l2(args.l2), name = 'fusion_layer_4'))
	fusion_model.add(Dense(1, weights=[np.array([[b]]),np.array([-a])], name = 'fusion_layer_5', trainable=False))
else:
	merged = Merge([Covarep_model, Facet_model, text_model], mode='concat', name = 'fusion_layer_0')
	fusion_model = Sequential()
	fusion_model.add(merged)
	fusion_model.add(Dropout(args.dropout)) #0.15
	fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(0.01), name = 'fusion_layer_1'))
	fusion_model.add(Dense(args.units, activation='relu', W_regularizer=l2(0.01), name = 'fusion_layer_2'))
	fusion_model.add(Dense(1, activation='sigmoid',W_regularizer=l2(0.01),))
	fusion_model.add(Dense(1, weights=[np.array([[b]]),np.array([-a])], trainable=False))
	
#fusion_model.load_weights(weights_folder_path + "fusion-pretrained-cv" + str(cv_id) + '.h5', by_name=True)
callbacks = [
	EarlyStopping(monitor='val_loss', patience=train_patience, verbose=0),
	ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]
sgd = SGD(lr=args.lr, decay=1e-6, momentum=args.momentum, nesterov=True)
adam = optimizers.Adamax(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #decay=0.999)
optimizer = {'sgd': sgd, 'adam':adam}
fusion_model.compile(loss=args.loss, optimizer=optimizer[args.optimizer])


# # check time
# startTime = datetime.now()
# predictions = fusion_model.predict([np.ones(X_Covarep_test.shape[0]), X_Covarep_test, X_Facet_test, X_text_test], verbose=0)
# timeElapsed=datetime.now()-startTime 
# print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
# assert False


#print(model.summary())
if use_cartesian_fusion:
	fusion_model.fit([np.ones(X_Covarep_train.shape[0]), X_Covarep_train, X_Facet_train, X_text_train], y_train, validation_split=val_split, nb_epoch=train_epoch, batch_size=args.batch, callbacks=callbacks)
	fusion_model.load_weights(weights_path)
	predictions = fusion_model.predict([np.ones(X_Covarep_test.shape[0]), X_Covarep_test, X_Facet_test, X_text_test], verbose=0)
else:
	fusion_model.fit([X_Covarep_train, X_Facet_train, X_text_train], y_train, validation_split=val_split, nb_epoch=train_epoch, batch_size=128, callbacks=callbacks)
	fusion_model.load_weights(weights_path)
	predictions = fusion_model.predict([X_Covarep_test, X_Facet_test, X_text_test], verbose=0)

predictions = predictions.reshape((len(y_test),))
y_test = y_test.reshape((len(y_test),))
mae = np.mean(np.absolute(predictions-y_test))
print "mae: ", mae
print "corr: ", round(np.corrcoef(predictions,y_test)[0][1],5)
print 'mult_acc: ', round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
true_label = (y_test >= 0)
predicted_label = (predictions >= 0)
print "Confusion Matrix :"
print confusion_matrix(true_label, predicted_label)
print "Classification Report :"
print classification_report(true_label, predicted_label, digits=5)
print "Accuracy ", accuracy_score(true_label, predicted_label)


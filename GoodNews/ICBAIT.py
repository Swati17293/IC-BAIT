import os
import csv
import pandas
import random
import shutil
import argparse
import numpy as np

import tensorflow as tf
import keras

from keras.models import *

from keras import metrics, Sequential
from keras.layers import *
from keras import optimizers

from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import *
from keras.preprocessing import text

import warnings
from sklearn.metrics import classification_report,accuracy_score

from sentence_transformers import SentenceTransformer

# Set a seed value
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value) 

def text_vectorization(split,dataset,embd,embd_model):

    model = SentenceTransformer(embd_model)
    model_sum = SentenceTransformer(embd_model)

    colnames = ['title','title_pre','summary','bias']
    df = pandas.read_csv(dataset+'/' + split + '.csv', names=colnames, sep='\t')

    lst_title = df.title_pre.tolist()
    lst_summary = df.summary.tolist()
    
    #Title vectorization..
    embeddings_title = model.encode(lst_title) 
    np.save(split + '_title_'+ embd + '_' + dataset + '.npy', embeddings_title)

    #Summary vectorization..
    embeddings_summary = model_sum.encode(lst_summary) 
    np.save(split + '_summary_'+ embd + '_' + dataset + '.npy', embeddings_summary)

    
def train_save_model(dataset,embd):

    dic = 3

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open(dataset +'/train.csv', 'r').readlines())
    num_valid = len(open(dataset +'/valid.csv', 'r').readlines())
    num_test = len(open(dataset +'/test.csv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_valid  : ' + str(num_valid) + ',  num_test  : ' + str(num_test) + '\n')
    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    input_shape = 768

    if embd == 'MiniLMv2':
        input_shape = 384

    encode_title = Input(shape=(input_shape,))

    encode_summary = Input(shape=(input_shape,))

    encode_summary_act = Activation('sigmoid')(encode_summary) 
    encode_summary_mul = Multiply()([encode_summary_act,encode_summary])
    encode_summary_x = Dense(768, activation='relu')(encode_summary_mul) 
    encode_summary_x= Dropout(0.5)(encode_summary_x)
    encode_summary_x = Dense(256, activation='relu')(encode_summary_x) 
    encode_summary_x= Dropout(0.5)(encode_summary_x)

    concat_input = Concatenate()([encode_title, encode_summary_x])
    
    gate_model = Dense(3, activation='softmax')(concat_input)

    gate_model = Model(inputs=[encode_title,encode_summary], outputs=gate_model)
    gate_model.summary()
    
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'models/'+ embd + '/' + dataset + '/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, mode='min', restore_best_weights=True)
    callbacks_list = [checkpoint, early_stopping]

    
    if os.path.isfile('models/'+ embd + '/' + dataset + '/MODEL.h5') == False:

        colnames  =  ['title','title_pre','summary','bias']
        df_train = pandas.read_csv(dataset +'/train.csv', names=colnames, sep='\t')
        df_valid = pandas.read_csv(dataset +'/valid.csv', names=colnames, sep='\t')
        df_test = pandas.read_csv(dataset +'/test.csv', names=colnames, sep='\t')

        train_bias = df_train.bias.tolist()
        train_bias_list = []

        for item in train_bias:
            if item == 'Left':
                train_bias_list.append(0)
            elif item == 'Center':
                train_bias_list.append(1)
            else:
                train_bias_list.append(2)

        valid_bias = df_valid.bias.tolist()
        valid_bias_list = []

        for item in valid_bias:
            if item == 'Left':
                valid_bias_list.append(0)
            elif item == 'Center':
                valid_bias_list.append(1)
            else:
                valid_bias_list.append(2)

        trainans = to_categorical(train_bias_list, 3)
        validans = to_categorical(valid_bias_list, 3)

        trainque_feature = np.load('train_title_'+ embd + '_' + dataset + '.npy')
        validque_feature = np.load('valid_title_'+ embd + '_' + dataset + '.npy')
        testque_feature = np.load('test_title_'+ embd + '_' + dataset + '.npy')

        trainsum_feature = np.load('train_summary_'+ embd + '_' + dataset + '.npy')
        validsum_feature = np.load('valid_summary_'+ embd + '_' + dataset + '.npy')
        testsum_feature = np.load('test_summary_'+ embd + '_' + dataset + '.npy')

        gate_model.fit([trainque_feature,trainsum_feature], trainans, epochs=10, batch_size=128, validation_data=([validque_feature,validsum_feature], validans), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('models/'+ embd + '/' + dataset + '/MODEL.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('models/'+ embd + '/' + dataset + '/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('models/'+ embd + '/' + dataset + '/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('models/'+ embd + '/' + dataset + '/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testque_feature,testsum_feature])

    fp = open('models/'+ embd + '/' + dataset + '/test.ans', 'w')

    for h in range(num_test):
        if np.argmax(ans[h]) == 0:
            fp.write('Left\n')
        elif np.argmax(ans[h]) == 1:
            fp.write('Center\n')
        else:
            fp.write('Right\n')

    fp.close()

def evaluate(dataset,embd):

    warnings.filterwarnings("ignore", category=UserWarning)
    
    f_test = open(dataset +'/test.csv')

    lines_test = f_test.readlines()

    true_ans_test = []

    for line in lines_test:
        bias = line.split('\t')[3].strip()
        true_ans_test.append(bias)

    f = open('models/'+ embd + '/' + dataset + '/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(true_ans_test, pred_ans)) 

    print('\n\n')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--embd", type=str, help="<Optional> Set the baseline model to be used in the learning framework. Options: ALBERT, DistilRoBERTa, MPNet, MiniLMv2, CMLM, GPL, GPLBPR, LaBSE", default='CMLM')
    args = parser.parse_args()

    embd = args.embd
    dataset = 'GoodNews'

    embd_model = 'sentence-transformers/use-cmlm-multilingual'
    
    if embd == 'ALBERT':
        embd_model = 'sentence-transformers/paraphrase-albert-base-v2'
    elif embd == 'DistilRoBERTa':
        embd_model = 'sentence-transformers/all-distilroberta-v1'
    elif embd == 'MPNet':
        embd_model = 'sentence-transformers/all-mpnet-base-v1'
    elif embd == 'MiniLMv2':
        embd_model = 'sentence-transformers/all-MiniLM-L12-v1'
    elif embd == 'CMLM':
        embd_model = 'sentence-transformers/use-cmlm-multilingual'
    elif embd == 'GPL':
        embd_model = 'GPL/trec-news-tsdae-msmarco-distilbert-gpl'
    elif embd == 'GPLBPR':
        embd_model = 'income/bpr-gpl-trec-news-base-msmarco-distilbert-tas-b'
    elif embd == 'LaBSE':
        embd_model = 'sentence-transformers/LaBSE'

    try:

        if os.path.isfile('test_title_' + embd + '_' + dataset + '.npy') == False:

            print('\n\nTurning text into vectors...')

            splits = ['train','valid','test']

            for split in splits:

                print('\nTurning ' + split + ' text into vectors...')
                text_vectorization(split,dataset,embd,embd_model)
                print('\n' + split + ' vectorization complete')
        
            print('\nVectorization complete...\n\n')
    except:
        pass

    if os.path.exists('models/' + embd + '/' + dataset + '/') == False:
        os.mkdir('models/' + embd + '/' + dataset + '/')
        train_save_model(dataset,embd)
        evaluate(dataset,embd)
        shutil.rmtree('models/' + embd + '/' + dataset + '/', ignore_errors=True)

if __name__ == "__main__":
    main()
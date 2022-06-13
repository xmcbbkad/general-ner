import pickle
import numpy
import os
import datetime
import keras
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, LSTM, Dropout, Embedding, TimeDistributed, Bidirectional
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
from keras_contrib.layers import CRF
import keras.backend.tensorflow_backend as KTF
from collections import OrderedDict
import ner_evaluate

DO_TRAIN=True
DO_EVAL=False
DO_PREDICT=True

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
session = tf.Session(config = config)
KTF.set_session(session)

SENTENCE_MAX_LENGTH = 128
CHAR_EMBEDDING_LENGTH = 200

#TRAIN_FILE = "../data/ontonotes_data/ontonotes-data_full_category_version_all.train"
#DEV_FILE = "../data/ontonotes_data/ontonotes-data_full_category_version_all.dev"
#TEST_FILE = "../data/ontonotes_data/ontonotes-data_full_category_version_all.test"
TRAIN_FILE = "../data/ontonotes_data/data_train"
DEV_FILE = "../data/ontonotes_data/data_dev"
TEST_FILE = "../data/ontonotes_data/data_test"
#TEST_FILE = "input_test_1"

DICT_CHAR = "dictionary.dict"
ARRAY_CHAR_EMBEDDING = "embedding_matrix.dict"


dict_char = pickle.load(open(DICT_CHAR, 'rb'))
array_char_embedding = pickle.load(open(ARRAY_CHAR_EMBEDDING, 'rb'))
VOCAB_SIZE = array_char_embedding.shape[0]
  
MODEL_FILE = 'model/model_ner_bilstm_crf.h5'
OUTPUT_DIR = 'output/'

LIST_LABEL = ['O','B-PER', 'I-PER', 'B-TIME', 'I-TIME', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-NORP', 'I-NORP', 'B-GPE', 'I-GPE', 'B-DEV', 'I-DEV',  'B-PERCENT', 'I-PERCENT', 'B-CUR', 'I-CUR', 'B-QUANTITY', 'I-QUANTITY', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK', 'I-WORK', 'B-LAN', 'I-LAN', 'B-LAW', 'I-LAW', 'B-ORDINAL', 'I-ORDINAL']

NUM_LABEL = len(LIST_LABEL)

MAP_LABEL_ID = OrderedDict()
MAP_ID_LABEL = OrderedDict()
for index,item in enumerate(LIST_LABEL):
    MAP_LABEL_ID[item] = index
    MAP_ID_LABEL[index] = item



def convert_input_data(file): 
    f = open(file, 'r')    

    list_char_all = []
    list_id_all = []
    list_y_all = []

    list_char = []
    list_id = []
    list_y = []

    for line in f:
        if line.strip() == '':
            if len(list_id) != len(list_y):
                print("test data process fail")
            if len(list_id) > SENTENCE_MAX_LENGTH:
                list_char = list_char[:SENTENCE_MAX_LENGTH]
                list_id = list_id[:SENTENCE_MAX_LENGTH]
                list_y = list_y[:SENTENCE_MAX_LENGTH]
            while len(list_id) < SENTENCE_MAX_LENGTH:
                list_char.append('@')
                list_id.append(0)
                null_id = [1]
                null_id.extend([0]*(NUM_LABEL-1))
                list_y.append(null_id)  
            
            list_char_all.append(list_char)
            list_id_all.append(list_id)
            list_y_all.append(list_y)        

            list_char = []
            list_id = []
            list_y = []         
        else:
            #print(line)
            char, y = line.rstrip().split('\t')
            list_char.append(char)
            if char in dict_char:
                list_id.append(numpy.float64(dict_char[char]))
            else:
                list_id.append(0)

            list_label_onehot = [0] * NUM_LABEL
            list_label_onehot[MAP_LABEL_ID[y]] = 1
            list_y.append(list_label_onehot)

    array_id_all = numpy.array(list_id_all)
    array_y_all = numpy.array(list_y_all)
    
    #array_id_all = array_id_all.astype(numpy.float32)
    #array_y_all = array_y_all.astype(numpy.float16)

    f.close()

    return list_char_all, array_id_all, array_y_all
'''
print('=== load train data & label ===')
x_char_train, x_train, y_train = convert_input_data(TRAIN_FILE)
print('=== load dev data & label ===')
x_char_dev, x_dev, y_dev = convert_input_data(DEV_FILE)
print('=== load test data & label ===')
x_char_test, x_test, y_test = convert_input_data(TEST_FILE)
'''

print('=== building network ===\n')
inputs = Input(shape = (SENTENCE_MAX_LENGTH,))
mid_layer = Embedding(VOCAB_SIZE, CHAR_EMBEDDING_LENGTH, weights = [array_char_embedding], input_length = SENTENCE_MAX_LENGTH, trainable = False)(inputs)
mid_layer = Bidirectional(LSTM(units = 128, return_sequences = True))(mid_layer)
mid_layer = Dropout(0.5)(mid_layer)
mid_layer = Bidirectional(LSTM(units = 64, return_sequences = True))(mid_layer)
mid_layer = Dropout(0.5)(mid_layer)
mid_layer = TimeDistributed(Dense(NUM_LABEL))(mid_layer)
crf = CRF(NUM_LABEL)
outputs = crf(mid_layer)
 
model = Model(inputs = inputs, outputs = outputs)    
model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

############################################################################## 
# 下述代码用于图像化训练结果，主要应用TensorBoard以及EarlyStopping模块：
# EarlyStopping模块：用于提前终止训练过程；
# TensorBoard模块：用于图形化测试结果，使用方法如下：
#     step 01: 在命令行语句中打入tensorboard --logdir=route(如 ./cys_logs) --port 6001(设置端口) --host 0.0.0.0(表示公开数据，可通过其他电脑访问)
#     step 02: 在浏览器中输入：http://192.168.3.11:6001(即端口名)即可查看
##############################################################################


if DO_TRAIN:
    print('=== load train data & label ===')
    present_time = '{}'.format(datetime.datetime.now().replace(microsecond=0).isoformat())
    tbCallback = TensorBoard(log_dir = ('tb_logs/ner_model_{}').format(present_time))
    esCallback = EarlyStopping(patience=3)
    
    x_char_train, x_train, y_train = convert_input_data(TRAIN_FILE)
    print('=== load dev data & label ===')
    x_char_dev, x_dev, y_dev = convert_input_data(DEV_FILE)
    print('=== training ===')
    for i in range(20):
        print('==========epoch={0} train'.format(i))
        
        model.fit(x_train, y_train, batch_size=500, epochs=1, callbacks = [esCallback, tbCallback])
    
        y_pred = model.predict(x_dev)
        score = ner_evaluate.my_F1(y_dev, y_pred)
        print('the accuracy is: %f' % score[0])
        print('the macro_F1 is: %f' % score[1])
        print('the micro_F1 is: %f' % score[2])

    #model.fit(x_train, y_train, batch_size=500, epochs=20, callbacks = [esCallback, tbCallback])

    model.save(MODEL_FILE)

if DO_EVAL:
    print('=== load dev data & label ===')
    x_char_dev, x_dev, y_dev = convert_input_data(DEV_FILE)
    
    #model = load_model(MODEL_FILE, custom_objects={"CRF":CRF, 'crf_loss':crf_loss, 'crf_viterbi_accuracy':crf_viterbi_accuracy})
    model = load_model(MODEL_FILE, custom_objects={"CRF":CRF, 'crf_loss':crf.loss_function, 'crf_viterbi_accuracy':crf.accuracy})
    print('\n\t=== the score in valid dataset ===')
    score = model.evaluate(x_dev, y_dev)
    print('\nthe loss is: ' + str(score[0]))
    print('the accuracy is: ' + str(score[1]))

    y_pred = model.predict(x_dev)
    ner_evaluate.my_evaluate(y_dev, y_pred)

if DO_PREDICT:
    print('=== load test data & label ===')
    x_char_test, x_test, y_test = convert_input_data(TEST_FILE)

    model = load_model(MODEL_FILE, custom_objects={"CRF":CRF, 'crf_loss':crf.loss_function, 'crf_viterbi_accuracy':crf.accuracy})
   
    #print('\n\t=== the score in test dataset ===')
    #score = model.evaluate(x_test, y_test)
    #print('\nthe loss is: ' + str(score[0]))
    #print('the accuracy is: ' + str(score[1]))
    y_pred = model.predict(x_test)
    
    #ner_evaluate.my_evaluate(y_test, y_pred)
 
    writer = tf.gfile.GFile(OUTPUT_DIR+"test_results.tsv", 'w')
    for i in range(len(x_char_test)):
        for j in range(len(x_char_test[i])):
            if x_char_test[i][j] == '@': break
            y_label_str = MAP_ID_LABEL[numpy.argmax(y_test[i][j])]
            y_pred_str = MAP_ID_LABEL[numpy.argmax(y_pred[i][j])]
            output_line = '{}\t{}\t{}\n'.format(x_char_test[i][j], y_label_str, y_pred_str) 
            writer.write(output_line)
        writer.write('\n')


keras.backend.clear_session()

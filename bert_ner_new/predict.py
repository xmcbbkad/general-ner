# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import json
from tqdm import tqdm

from bert import tokenization
from model import BertNer

import common

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'predict', 'Must be one of train/predict')
flags.DEFINE_string('vocab_path', 'chinese_L-12_H-768_A-12/vocab.txt', 'Path of the vocab file')
flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
flags.DEFINE_string('test_file_path', '', 'Path of the test file')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
flags.DEFINE_integer('max_seq_length', 128, 'Max length of the sequence')

LABEL_LIST = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME', 'B-NORP', 'I-NORP', 'B-GPE', 'I-GPE', 'B-DEV', 'I-DEV',  'B-PERCENT', 'I-PERCENT', 'B-CUR', 'I-CUR', 'B-QUANTITY', 'I-QUANTITY', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK', 'I-WORK', 'B-LAN', 'I-LAN', 'B-LAW', 'I-LAW', 'B-ORDINAL', 'I-ORDINAL']

MAP_ID_LABEL = {}
for (i, label) in enumerate(LABEL_LIST):
    MAP_ID_LABEL[i] = label





def main(argv):
    hparams_path = os.path.join(FLAGS.log_root, 'hparams.json')
    with open(hparams_path, 'r') as result_file:
        h_json = json.load(result_file)
    h_json = json.loads(h_json)
    h_json['mode'] = FLAGS.mode
    hparams = tf.contrib.training.HParams()
    for key,value in h_json.items():
        hparams.add_hparam(key,value)

    predict_log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(predict_log_root):
        os.makedirs(predict_log_root)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_path, do_lower_case=False)

    print('Start to load the test data...')
    test_data = common.create_examples(FLAGS.test_file_path)
    print('Test data loaded')

    test_data_batch = common.generate_batch(test_data, FLAGS.batch_size, FLAGS.max_seq_length, tokenizer, False)

    model = BertNer(hps = hparams)
    
    writer = tf.gfile.GFile('log/predict_result.txt', 'w')

    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_root))
            
            for eval_input_raw, eval_input_ids_list, eval_input_mask_list, eval_segment_ids_list, eval_pred_ids_list in tqdm(test_data_batch):
                eval_input_ids_list = np.asarray(eval_input_ids_list,dtype = np.int32)
                eval_input_mask_list = np.asarray(eval_input_mask_list,dtype = np.int32)
                eval_segment_ids_list = np.asarray(eval_segment_ids_list,dtype = np.int32)
                eval_pred_ids_list = np.asarray(eval_pred_ids_list,dtype = np.int32)

                predictions = model.predict(eval_input_ids_list, eval_input_mask_list, eval_segment_ids_list, sess)
                
                #for i in range(FLAGS.batch_size):
                for i in range(len(eval_input_ids_list)):
                    char_list = tokenizer.convert_ids_to_tokens(eval_input_ids_list[i])
                    for j in range(FLAGS.max_seq_length):
                        if eval_input_mask_list[i][j] == 1:
                            line = "{}\t{}\t{}\n".format(char_list[j], MAP_ID_LABEL[eval_pred_ids_list[i][j]], MAP_ID_LABEL[predictions[i][j]])
                            writer.write(line)
                    writer.write('\n')
                    writer.flush()
            writer.close()        


if __name__ == "__main__":
    tf.app.run()


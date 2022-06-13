# -*- coding: utf-8 -*-
import sys
import json
import tensorflow as tf
from simplex_sdk import SimplexClient


def call_baidu_api(euler_client, str_input):
    ret = euler_client.predict(str_input)
    predict_y = ['O'] * len(str_input)
    json_str = euler_client.predict(str_input)
    if(not isinstance(json_str, str)):
        return predict_y

    json_dict = json.loads(json_str)

    for item in json_dict['output_ner']:
        predict_y[item['offset']] = 'B-{}'.format(item['tag'])
        for i in range(item['length']-1):
            predict_y[item['offset']+1+i] = 'I-{}'.format(item['tag'])
   
    output_char = list(json_dict['bd_input_text'])

    return output_char, predict_y


if __name__ == '__main__':
    f_predict = open(sys.argv[1], 'r')

    euler_client = SimplexClient('NerBaiduApi', namespace='dev')

    list_char = []
    list_y = []
    list_pred = []

    writer = tf.gfile.GFile(sys.argv[2], 'w')
    count_batch = 0
    lines = f_predict.readlines()
    for i in range(len(lines)):
        line = lines[i]

        if(i % 100 == 0):
            print(i)
     
        if line.strip() == '':
            list_char.append('\n')
            list_y.append('O')

            if(count_batch > 4000 or i == len(lines) - 1):
                text = ''.join(list_char)
                list_output_char, list_pred = call_baidu_api(euler_client, text)
                count_batch = 0
                if(len(list_output_char) != len(list_char)):
                    print("input_char_len:{} output_char_len:{}".format(len(list_char), len(list_output_char)))
                for j in range(len(list_output_char)):
                    output_line = "{}\t{}\n".format(list_output_char[j], list_pred[j])
                    if(list_output_char[j] == '\n'):
                        output_line = '\n'
                    elif(len(list_y) == len(list_pred)):
                        output_line = "{}\t{}\t{}\n".format(list_output_char[j], list_y[j], list_pred[j])
                    writer.write(output_line)
                writer.flush()
                list_char = []
                list_y = []

        else:
            list_line = line.strip().split()
            list_char.append(list_line[0])
            if(len(list_line) == 2):
                list_y.append(list_line[1])
        count_batch += 1

    writer.close()

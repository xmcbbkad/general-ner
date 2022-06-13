# -*- coding: utf-8 -*-
import sys
import json
import tensorflow as tf
from bosonnlp import BosonNLP

API_TOKEN = "t_4yIYq2.34681.-r-tXXznW48M"


MAP_NAME={"location":"LOC", "time":"TIME", "person_name":"PER", "org_name":"ORG", "company_name":"ORG", "product_name":"DEV"}

def NameNormalize(str_before):
    if str_before in MAP_NAME:
        return MAP_NAME[str_before]
    else: 
        return "O"



def call_boson_api(boson_client, str_input):
    ret = boson_client.ner(str_input)
    #print(ret)
    list_output_char = []
    list_output_pred = []

    word_list = ret[0]["word"]
    tag_list = ret[0]["entity"]
    
    word_idx = 0
    entity_idx = 0
    
    while word_idx < len(word_list):
        if entity_idx < len(tag_list) and word_idx == tag_list[entity_idx][0]:
            list_entity = ''.join(word_list[tag_list[entity_idx][0]: tag_list[entity_idx][1]])
            list_output_char.append(list_entity[0])
            if tag_list[entity_idx][2] in MAP_NAME: 
                list_output_pred.append("B-{}".format(NameNormalize(tag_list[entity_idx][2])))
            else:
                list_output_pred.append("O")
            #print ("{}\tB-{}".format(list_entity[0], NameNormalize(tag_list[entity_idx][2])))
            for i in range(1,len(list_entity)):
                list_output_char.append(list_entity[i])
                if tag_list[entity_idx][2] in MAP_NAME: 
                    list_output_pred.append("I-{}".format(NameNormalize(tag_list[entity_idx][2])))
                else:
                    list_output_pred.append("O")
                #print("{}\tI-{}".format(list_entity[i], NameNormalize(tag_list[entity_idx][2])))
            word_idx = tag_list[entity_idx][1]
            entity_idx +=1
        else:
            for item in list(word_list[word_idx]):
                list_output_char.append(item)
                list_output_pred.append("O")
                #print("{}\tO".format(item))
            word_idx += 1

    return list_output_char, list_output_pred
        




if __name__ == '__main__':
    f_predict = open(sys.argv[1], 'r')

    boson_client = BosonNLP(API_TOKEN)

    list_char = []
    list_y = []
    list_pred = []

    writer = tf.gfile.GFile(sys.argv[2], 'w')
    count_batch = 0
    lines = f_predict.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if(i%1000 == 0):
            print(i)
        
        if line.strip() == '':
            list_char.append('')
            list_y.append('O')
            if(count_batch > 4000 or i == len(lines) - 1):
                text = ''.join(list_char)
                list_output_char, list_pred = call_boson_api(boson_client, text) 
                count_batch = 0
                if(len(list_output_char) != len(list_char)):
                    print("input_char_len:{} output_char_len:{}".format(len(list_char), len(list_output_char)))
                for j in range(len(list_output_char)):
                    output_line = "{}\t{}\n".format(list_output_char[j], list_pred[j])
                    if(list_output_char[j] == ''):
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

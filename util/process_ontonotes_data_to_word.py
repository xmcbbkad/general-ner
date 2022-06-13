# -*- coding: utf-8 -*-
# ontonotes raw data --> char+\t+tag(chinese)  word+\t+tag(english)

import re
import os
import sys
sys.path.append('../')
from bert_ner.bert import tokenization

basic_tokenizer = tokenization.BasicTokenizer(do_lower_case = False)

def should_split(last_char, char):
    if last_char == '' or char == '':
        return False
    
    if tokenization._is_punctuation(last_char) or tokenization._is_punctuation(char):
        return True

    if basic_tokenizer._is_chinese_char(ord(last_char)) or basic_tokenizer._is_chinese_char(ord(char)):
        return True

    return False

def write_into_file(word, tag, fout, state):
    #print(word)

    DICT_eff = {'PERSON':'PER', 'NORP':'NORP', 'FAC':'LOC', 'ORG':'ORG', 'GPE':'GPE', 'LOC':'LOC',
                'PRODUCT':'DEV', 'DATE':'TIME', 'TIME':'TIME', 'PERCENT':'PERCENT', 'MONEY':'CUR', 
                'QUANTITY':'QUANTITY', 'ORDINAL':'ORDINAL', 'CARDINAL':'CARDINAL', 'EVENT':'EVENT',
                'WORK_OF_ART':'WORK', 'LAW':'LAW', 'LANGUAGE':'LAN'}
    if tag in DICT_eff.keys():
        label = DICT_eff[tag]
    else:
        label = 'O'
    count = 0
    
    #tokens = list(word)
    #tokens = tokenizer.tokenize(word)    
    #tokens = []
    #tokens.append(word)
    #for char in tokens:
    #for char in basic_tokenizer.tokenize(word):

    tmp_chars = ''
    last_char = ''

    for i in range(len(word)):
        if should_split(last_char, word[i]) == False:
            tmp_chars += word[i]
            last_char = word[i]

            continue

        if label == 'O':
            fout.write('%s\t%s\n' % (tmp_chars, 'O'))
        else:
            if count == 0 and state:
                fout.write('%s\t%s\n' % (tmp_chars, 'B-'+label))
            else:
                fout.write('%s\t%s\n' % (tmp_chars, 'I-'+label))
        count += 1
        tmp_chars = word[i]
        last_char = word[i]

    if tmp_chars != '':
        if label == 'O':
            fout.write('%s\t%s\n' % (tmp_chars, 'O'))
        else:
            if count == 0 and state:
                fout.write('%s\t%s\n' % (tmp_chars, 'B-'+label))
            else:
                fout.write('%s\t%s\n' % (tmp_chars, 'I-'+label))
 
                
    '''
    for char in list(word):
        if label == 'O':
            fout.write('%s\t%s\n' % (char, label))
        else:
            if count == 0 and state:
                fout.write('%s\t%s\n' % (char, 'B-'+label))
            else:
                fout.write('%s\t%s\n' % (char, 'I-'+label))
        count += 1
    
    
    #if(len(list(word)) > 0):
    #    if basic_tokenizer._is_chinese_char(ord(list(word)[-1])) == False:
    #        fout.write('%s\t%s\n' % (' ', 'O'))    

    return True
    '''
##########################################################################
# change_format()函数将fin_path指向的原始数据文件转换为可以识别的文件格式，
# 并将其续写到fout_path所指向的文件中。
##########################################################################
def change_format(fin_path, fout_path):
    
    NER_PATTERN = re.compile('^TYPE="(.*?)"(?:[ES]_OFF="\d")?>(.*?)</ENAMEX>$')
    NER_BEGIN_PATTERN = re.compile('^TYPE="(.*?)">(.*?)$')
    NER_END_PATTERN = re.compile('^(.*?)</ENAMEX>$')
    NER_EXCEPTION = re.compile('^TYPE="(.*?)"$')
    NER_PATTERN_VER2 = re.compile('^[ES]_OFF="\d">(.*?)</ENAMEX>$')
    NER_BEGIN_PATTERN_VER2 = re.compile('^[ES]_OFF="\d">(.*?)$')

    fin = open(fin_path,'r')
    fout = open(fout_path,'a+')

    count = 0
    for line in fin:
        flag = True
        s = line.strip().split(' ')
        lock = False
        tag = 'O'
        for token in s:
            # print(token,end = '\t')
            if token == '<DOC' or token == '</DOC>' or token == 'ＥＭＰＴＹ':
                flag = False
                break
            elif token == '<ENAMEX':
                continue
            elif NER_PATTERN.match(token):
                tag = NER_PATTERN.match(token).group(1)
                word = NER_PATTERN.match(token).group(2)
                # print('%s\t%s' % (word,tag))
                write_into_file(word, tag, fout, True)
            elif NER_EXCEPTION.match(token):
                tag = NER_EXCEPTION.match(token).group(1)
            elif NER_PATTERN_VER2.match(token):
                word = NER_PATTERN_VER2.match(token).group(1)
                # print(word,end = '\t')
                # print(tag)
                write_into_file(word, tag, fout, True)
            elif NER_BEGIN_PATTERN_VER2.match(token):
                word = NER_BEGIN_PATTERN_VER2.match(token).group(1)
                # print(word,end = '')
                write_into_file(word, tag, fout, True)
                lock = True
            elif NER_BEGIN_PATTERN.match(token):
                tag = NER_BEGIN_PATTERN.match(token).group(1)
                word = NER_BEGIN_PATTERN.match(token).group(2)
                # print(word,end = '')
                write_into_file(word, tag, fout, True)
                lock = True
            elif NER_END_PATTERN.match(token):
                word = NER_END_PATTERN.match(token).group(1)
                # print(word,end = '\t')
                # print(tag)
                write_into_file(word, tag, fout, False)
                lock = False
            else:
                if lock == True:
                    # print(token,end='')
                    write_into_file(token, tag, fout, False)
                else:
                    tag = 'O'
                    write_into_file(token, tag, fout, False)
        if flag:
            fout.write('\n')
        # print(s)
        count += 1
        if count % 100 == 0:
            print('=== %d sentence changed ===' % count)
    fin.close()
    fout.close()
    
    return True

####################################################################################
# traverse_datafile()函数用于遍历rootpath文件夹内的文件，
# 并找出其中后缀名为suffix的文件，对其进行格式转换并写入到fout_path路径指向的文件：
####################################################################################
def traverse_datafile(rootpath, suffix, fout_path):
    
    FILENAME_PATTERN = re.compile('^(.*?)' + suffix + '$')
    
    files = os.listdir(rootpath)
    for file in files:
        # print(file)
        if os.path.isdir(rootpath + '/' + file):
            # print(rootpath + '/' + file)
            traverse_datafile(rootpath + '/' + file, suffix, fout_path)
        else:
            if FILENAME_PATTERN.match(file):
                print(rootpath + '/' + file)
                change_format(rootpath + '/' + file, fout_path+file)
    
    return True

'''
==================================
=== 这一部分为实际的运行函数模块 ===
==================================
'''
if __name__ == '__main__':
    # === rootpath为ontonotes-release-5.0数据文件中所有的中文标注文件的存储目录根路径 ===
    #rootpath = '/data/share/ontonotes-release-5.0/data/files/data/chinese/annotations'
    rootpath = '/data/share/ontonotes-release-5.0/data/files/data/english/annotations'
    #rootpath = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/test'
    #rootpath = '/data/share/ontonotes-release-5.0/data/files/data/english/annotations/tc/ch/00'
    
    #fout_path = './data/ontonotes_data/ontonotes_chinese_basic_tokenize/'
    #fout_path = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/english_test_1/'
    #fout_path = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/chinese_test_1/'
    fout_path = '/data/jh/notebooks/fanxiaokun/code/general_ner/data/ontonotes_data/english_word_20190714/'

    suffix = '.name'
    traverse_datafile(rootpath, suffix, fout_path)

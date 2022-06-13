# -*- coding: utf-8 -*-
import sys

RESERVE_LIST = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME', 'B-DEV', 'I-DEV']
RESERVE_LIST = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
#RESERVE_LIST = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME']

REPLACE_MAP = {'I-GPE':'I-LOC', 'B-GPE':'B-LOC', 'I-NORP':'I-LOC', 'B-NORP':'B-LOC'}

f_in = open(sys.argv[1],'r', encoding='utf-8')
f_out = open(sys.argv[2], 'w')

for line in f_in:
    if(line == '\n'):
        f_out.write('\n')
    else:
        char = ''
        y_true = ''
        y_pred = ''
        item = line.strip().split()
        char = item[0]
        if(len(item) == 2):
            y_pred = item[1]
            if y_pred in REPLACE_MAP:
                y_pred = REPLACE_MAP[y_pred]
            if y_pred not in RESERVE_LIST:
                y_pred = 'O'
            f_out.write("{}\t{}\n".format(char, y_pred))
        elif(len(item) == 3): 
            y_true = item[1]
            y_pred = item[2]
            if y_true in REPLACE_MAP:
                y_true = REPLACE_MAP[y_true]
            if y_pred in REPLACE_MAP:
                y_pred = REPLACE_MAP[y_pred]
            if y_true not in RESERVE_LIST:
                y_true = 'O'
            if y_pred not in RESERVE_LIST:
                y_pred = 'O'
            f_out.write("{}\t{}\t{}\n".format(char, y_true, y_pred))

f_in.close()
f_out.close()
            
    

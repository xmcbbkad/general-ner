# -*- coding: utf-8 -*-

import sys

f_in = open(sys.argv[1],'r', encoding='utf-8')
f_out = open(sys.argv[2], 'w')

for line in f_in:
    tokens = line.strip().split()
    #for char in list(line):
    for char in tokens:
        if(char == '\n'):
            f_out.write('\n')
        else:
            f_out.write("{}\t{}\n".format(char, 'O'))

f_in.close()
f_out.close()

# -*- coding: utf-8 -*-

import sys

f_in = open(sys.argv[1],'r', encoding='utf-8')
f_out = open(sys.argv[2], 'w')

for line in f_in:
    for char in list(line):
        f_out.write(char)
        if(char != '\n' and char !='\r'):
            f_out.write('\n')
f_in.close()
f_out.close()

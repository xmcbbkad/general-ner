# -*- coding: utf-8 -*-

import sys

f_in = open(sys.argv[1],'r', encoding='utf-8')
f_out = open(sys.argv[2], 'w')

lines = f_in.readlines()

for i in range(len(lines)):
    line = lines[i]
    last_line = ''
    if i != 0:
        last_line = lines[i-1]
    
    if last_line.strip() == '':
         line = line.replace("I-", "B-")

    f_out.write(line)

f_in.close()
f_out.close()

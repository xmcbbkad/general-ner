# -*- coding: utf-8 -*-

import sys

f_in = open(sys.argv[1],'r', encoding='utf-8')
f_out = open(sys.argv[2], 'w')

output_text = ''
for line in f_in:
    if line.strip() == '':
        output_text += '\n'
    else:
        item = line.strip().split()
        output_text += item[0]

f_out.write(output_text)

f_in.close()
f_out.close()

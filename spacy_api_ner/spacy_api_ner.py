# -*- coding: utf-8 -*-
import sys,os
import spacy

NER_MAP = {"PERSON":"PER", "NORP":"NORP", "FAC":"LOC", "ORG":"ORG", "GPE":"GPE", "LOC":"LOC", "PRODUCT":"DEV", "EVENT":"EVENT", "WORK_OF_ART":"WORK", "LAW":"LAW", "LANGUAGE":"LAN", "DATE":"TIME", "TIME":"TIME", "PERCENT":"PERCENT", "MONEY":"CUR", "QUANTITY":"QUANTITY", "ORDINAL":"ORDINAL", "CARDINAL":"CARDINAL"}

nlp = spacy.load("en_core_web_sm")

class word():
    def __init__(self, text, begin_index, end_index, true_tag, pre_tag='O'):
        self.text = text
        self.begin_index = begin_index
        self.end_index = end_index
        self.true_tag = true_tag
        self.pre_tag = pre_tag


def nerPhraseToWord(ner_info):
    list_output = []

    for en in ner_info:
        begin_index = 0
        for i in range(len(en.text)):
            label = 'B-'+NER_MAP[en.label_]
            if begin_index != 0:
                label = 'I-'+NER_MAP[en.label_]
            if i == len(en.text) -1:
                list_output.append(word((en.text)[begin_index:i+1], en.start_char+begin_index, en.start_char+i+1, 'O', label)) 
            elif (en.text)[i] == ' ':
                list_output.append(word((en.text)[begin_index:i], en.start_char+begin_index, en.start_char+i, 'O', label))
                begin_index = i+1        
    return list_output

if __name__ == '__main__':
    f_input = open(sys.argv[1], 'r')
    f_output = open(sys.argv[2], 'w')

    lines = f_input.readlines()

    list_sentence_word = []
    sum_offset = 0

    for i in range(len(lines)):
        line = lines[i]
        if (i%10000==0):
            print("{}/{}".format(i, len(lines)))

        if line.strip() == '':
            text = ''
            for item in list_sentence_word:
                text += (' '+item.text)
            doc = nlp(text)

            #for en in doc.ents:
            #    print(en.text, en.label_)

            list_entity = nerPhraseToWord(doc.ents)
            for entity in list_entity:
                for item in list_sentence_word:
                    if entity.begin_index == item.begin_index:
                        item.pre_tag = entity.pre_tag
            
            for item in list_sentence_word:
                f_output.write("{}\t{}\t{}\n".format(item.text, item.true_tag, item.pre_tag))
            f_output.write("\n")

            list_sentence_word = []
            sum_offset = 0
        else:
            l = line.strip().split()
            list_sentence_word.append(word(l[0], sum_offset+1 , sum_offset+1+len(l[0]), l[1]))
            sum_offset += (1+len(l[0])) 

    f_input.close()
    f_output.close()

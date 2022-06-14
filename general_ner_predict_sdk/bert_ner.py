# -*- coding: utf-8 -*-
import os
import requests
import logging
import copy

from general_ner_predict_sdk import tokenization
from general_ner_predict_sdk import sentence

logger = logging.getLogger(__name__)

class BertNerApi():
    def __init__(self, *args, **kwargs):
        super(BertNerApi, self).__init__(*args, **kwargs)

        #self.vocab_file_path = self.download(kwargs['vocab_file_path'])
        #self.vocab_file = kwargs.get('vocab_file', 'multi_lang_vocab.txt')
        self.vocab_file = kwargs.get('vocab_file', 'chinese_vocab.txt')
        self.vocab_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", self.vocab_file)
        self.max_seq_length = kwargs.get('max_seq_length', 192)
        #self.max_seq_length = kwargs.get('max_seq_length', 128)
        self.do_lower_case = kwargs.get('do_lower_case', 1)
        self.full_tokenizer = tokenization.FullTokenizer(vocab_file = self.vocab_file_path, do_lower_case = self.do_lower_case)
        self.basic_tokenizer = tokenization.BasicTokenizer(do_lower_case = self.do_lower_case)
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab = tokenization.load_vocab(self.vocab_file_path))

        self.tf_host_str = kwargs.get("tf_host_str", "") #tf_serving url 

        LABEL_LIST = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME', 'B-NORP', 'I-NORP', 'B-GPE', 'I-GPE', 'B-DEV', 'I-DEV',  'B-PERCENT', 'I-PERCENT', 'B-CUR', 'I-CUR', 'B-QUANTITY', 'I-QUANTITY', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK', 'I-WORK', 'B-LAN', 'I-LAN', 'B-LAW', 'I-LAW', 'B-ORDINAL', 'I-ORDINAL']
        self.map_id2name = {}
        for (i, label) in enumerate(LABEL_LIST):
            self.map_id2name[i] = label

    def predict(self, data, **kwargs):
        instance_list = []
        sentence_tokens_list = []
        
        list_len_text = []
        for item in data:
            list_len_text.append(len(item))
            sentence_tokens = sentence.SentenceTokenizer(text=item, wordpiece_tokenizer=self.wordpiece_tokenizer, basic_tokenizer=self.basic_tokenizer, do_lower_case=self.do_lower_case, max_seq_length=self.max_seq_length)
            sentence_tokens_list.append(sentence_tokens)
            input_ids, input_mask, input_segment_ids = self.encode_single_input(sentence_tokens, self.max_seq_length)
            instance_list.append({"input_ids":input_ids, "input_mask":input_mask, "segment_ids":input_segment_ids})
        logger.info("batch_size={}, len={}".format(len(data), list_len_text))
        #print("batch_size={}, len={}".format(len(data), list_len_text))

        result = self.get_tf_results(instance_list)
        
        final_output_list = [] 
        
        if result == None:
            return final_output_list


        predictions = result['predictions']
        assert len(predictions) == len(data)
        assert len(predictions) == len(instance_list)
        
        for i in range(len(predictions)):
            output_item = self.parse_result(sentence_tokens_list[i], instance_list[i], predictions[i])
            final_output_list.append(output_item)

        return final_output_list


    def get_tf_results(self, instance_list):
        data = {"signature_name":"predict", "instances":instance_list}

        retry = 2
        while retry > 0:
            try:
                ret = requests.post(self.tf_host_str, json=data, timeout=100)
                if ret.status_code == 200:
                    return ret.json()
                else:
                    logger.critical("call tf fail")
                    retry -=1

            except:
                logger.critical("call tf fail")
                retry -= 1
        
        return None

    def encode_single_input(self, sentence_tokens, max_seq_length):
        input_ids = []
        input_mask = []
        input_segment_ids = []
        tokens = []
        
        tokens_raw = copy.copy(sentence_tokens.list_wordpiece)
        if len(tokens_raw) > max_seq_length - 2:
            tokens_raw = tokens_raw[:max_seq_length-2]
        
        tokens.append("[CLS]")
        input_segment_ids.append(0)
        for token in tokens_raw:
            tokens.append(token)
            input_segment_ids.append(0)
        tokens.append("[SEP]")
        input_segment_ids.append(0)
 
        input_ids = self.full_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_segment_ids.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_segment_ids) == max_seq_length

        return input_ids, input_mask, input_segment_ids 


    def encode_single_input_1(self, input_text, max_seq_length):
        #text_all = ''.join(input_text.split())
        text_unicode = tokenization.convert_to_unicode(input_text)
        #tokens_a = self.full_tokenizer.tokenize(text_unicode)
        tokens_b = list(text_unicode)
        tokens_a = []
        for item in tokens_b:
            tokens_a.extend(self.full_tokenizer.tokenize(item))

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length-2]
        tokens = []
        input_segment_ids = []
        tokens.append("[CLS]")
        input_segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_segment_ids.append(0)
        tokens.append("[SEP]")
        input_segment_ids.append(0)
        input_ids = self.full_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_segment_ids) == max_seq_length

        return input_ids, input_mask, input_segment_ids 

    def parse_result(self, sentence_tokens, encode_data, predict_data):
        dict_output = {}
        dict_output["raw_str"] = sentence_tokens.text_raw
        dict_output["normalized_str"] = ""
        dict_output["ne"] = []

        tokens = self.full_tokenizer.convert_ids_to_tokens(encode_data["input_ids"])
        dict_output["normalized_str"] =' '.join(tokens[1:])

        visited_idx = 1
        for i in range(1, len(tokens)):
            if i < visited_idx: continue
            visited_idx = i

            if tokens[i] == "[SEP]":
                dict_output["normalized_str"] = ' '.join(tokens[1:i])
                break
                
            if predict_data[i] == 0 or predict_data[i]%2 == 0: continue

            ner_begin = i
            ner_end = i
            ner_name = self.map_id2name[predict_data[i]][2:]
            
            for j in range(i+1, len(tokens)):
                ner_end = j
                if predict_data[j] != predict_data[i]+1 or tokens[j] == "[SEP]": 
                    break
            
            visited_idx = ner_end
            
            offset = sentence_tokens.group[sentence_tokens.list_wordpiece_group_index[ner_begin-1]].offset
            length = 0
            text = ''
            last_group_index = -1
            
            group_index_begin = sentence_tokens.group[sentence_tokens.list_wordpiece_group_index[ner_begin-1]].index
            group_index_end = sentence_tokens.group[sentence_tokens.list_wordpiece_group_index[ner_end-2]].index
            
            for k in range(group_index_begin, group_index_end+1):
                group_text = sentence_tokens.group[k]
                length += sentence_tokens.group[k].length
                text += sentence_tokens.group[k].text

            '''
            for k in range(ner_begin, ner_end):
                group_text = sentence_tokens.group[sentence_tokens.list_wordpiece_group_index[k-1]]
                if group_text.index != last_group_index:
                    last_group_index = group_text.index
                    length += group_text.length
                    text += group_text.text
            '''
            


            dict_output["ne"].append({"tag":ner_name, "offset":offset, "length":length, "text":text})
        
        return dict_output




    def parse_result_1(self, raw_data, encode_data, predict_data):
        dict_output = {}
        dict_output["raw_input_str"] = raw_data
        dict_output["ne"] = []

        tokens = self.full_tokenizer.convert_ids_to_tokens(encode_data["input_ids"])
        

        visited_idx = 1
        for i in range(1, len(tokens)):
            if i < visited_idx: continue
            visited_idx = i

            if tokens[i] == "[SEP]":
                dict_output["tokenize_input_str"] = ''.join(tokens[1:i])
                break
                
            if predict_data[i] == 0 or predict_data[i]%2 == 0: continue

            ner_begin = i
            ner_end = -1
            ner_name = self.map_id2name[predict_data[i]][2:]
            
            for j in range(i+1, len(tokens)):
                ner_end = j
                if predict_data[j] != predict_data[i]+1: 
                    break
            
            visited_idx = ner_end

            dict_output["ne"].append({"tag":ner_name, "offset":i-1, "length":ner_end-ner_begin, "text": "".join(tokens[ner_begin:ner_end])})
        
        return dict_output

class BertNerApi_tmp(BertNerApi):
    def __init__(self, *args, **kwargs):
        super(BertNerApi_tmp, self).__init__(*args, **kwargs)

        #self.vocab_file_path = self.download(kwargs['vocab_file_path'])
        self.vocab_file = kwargs.get('vocab_file', 'multi_lang_vocab.txt')
        #self.vocab_file = kwargs.get('vocab_file', 'chinese_vocab.txt')
        self.vocab_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", self.vocab_file)
        self.max_seq_length = kwargs.get('max_seq_length', 192)
        #self.max_seq_length = kwargs.get('max_seq_length', 128)
        #self.do_lower_case = kwargs.get('do_lower_case', 1)
        self.do_lower_case = kwargs.get('do_lower_case', 0)
        
        self.full_tokenizer = tokenization.FullTokenizer(vocab_file = self.vocab_file_path, do_lower_case = self.do_lower_case)
        self.basic_tokenizer = tokenization.BasicTokenizer(do_lower_case = self.do_lower_case)
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab = tokenization.load_vocab(self.vocab_file_path))

        self.tf_host_str = kwargs.get("tf_host_str", "")  #tfserving host


if __name__ == "__main__":
    model = BertNerApi()
    #model_1 = BertNerApi_tmp()
    input_text_list = []
    input_text_list.append("毛泽东  是国家主席,他生于湖南长沙，没去过美国。新华社北京6月14日电6月14日，“2019·中国西藏发展论坛”在西藏拉萨举行。国家主席习近平发来贺信，向论坛开幕表示祝贺。")
    #input_text_list = ["AMERICAN CTO 机器学习和自然语言处理专家，香港科技大学博士。曾任职于微软研究院、Bing 搜索，担任 Cortana 首席算法科学家并因此获得微软个人贡献奖。"]
    #input_text_list = ["班加罗尔共享出行公司Bounce募集7200万美元C轮融资，投资方包括Chiratae Ventures、Accel India、红杉资本印度和 Omidyar Network。"]
    #input_text_list = ["Mission accomplished? Trump's meeting with Kim is a political win despite long odds of diplomatic success"]
    
    result = model.predict(input_text_list)
    print(result)
    print('----------------')


# 标签体系:
```
KI_dbpedia类别,ontonotes类别,tag类别,标签含义
PERSON,Person,PER,人物
NORP,[NULL],NORP,政治/宗教团体
FAC,Place,LOC,人造建筑
ORG,Organisation,ORG,组织
GPE,[NULL],GPE,国家/城市/州
LOC,Place,LOC,地点
PRODUCT,Device,DEV,设备，产品
DATE,TimePeriod,TIME,时间
TIME,TimePeriod,TIME,时间
PERCENT,[NULL],PERCENT,百分比
MONEY,Currency,CUR,金钱
QUANTITY,[NULL],QUANTITY,数量
ORDINAL,[NULL],ORDINAL,序数词
CARDINAL,[NULL],CARDINAL,数词
EVENT,Event,EVENT,事件
WORK_OF_ART,Work,WORK,艺术作品
LAW,Law(包含于Work中),LAW/WORK,法律
LANGUAGE,Language,LAN,语言
```

# data：
## ontonotes:
1. 原始数据：ontonotes-release-5.0/

2. 处理程序1：util/process_ontonotes_data_to_word.py<br/>
    说明：把原始ontonotes数据处理成标准NER格式的数据，用BasicTokenizer切分，中文一个字为一个单位，英文一个单词为一个单位

3. 处理程序2：util/split_ontonotes_data.py<br/>
    说明：把处理程序1处理过的数据，分割为train/dev/test三份

4. 处理程序3：util/process_ontonotes_word_to_wordpiece.py<br/>
    说明：把原始ontonotes数据处理成标准NER格式的数据，用WordpieceTokenizer切分，适合bert使用。
    注意：Bert给定中文词表是不包括大写字母(A-Z)的，所以必须得do_lower_case. 英文词表包括大写字母，不用do_lower_case.



# model：
## BiLSTM+CRF:
cd bisltm_crf_ner/<br/>
训练：python ner_bilstm_crf.py<br/>
效果评估：见readme.txt<br/>
## BERT
cd bert_ner_new/<br/>
训练：sh run_bert_ner.sh<br/>
效果评估：见readme.txt<br/>
## baidu api:
见baidu_api_ner/readme.txt<br/>
## boson api:
见boson_api_ner/readme.txt<br/>



# util：
util/split_char.py<br/>
  功能：用于把连续文本切分为标准NER输入格式（一行一个字符）.<br/>
  用法: python split_char.py input_file output_file <br/>
  param:<br/>
    input_file：输入的正常文本，可以是多行。<br/>
    output_file：输出的标准NER格式。<br/>

util/merge_char.py<br/>
  功能：和split_char.py相反，把标准NER格式的文本转换为连续文本.<br/>
  param:<br/>
    input_file：输出的标准NER格式。<br/>
    output_file：连续文本。<br/>

util/change_tag.py<br/>
  功能：剔除掉api中没有定义的实体类型，合并同义实体类型（GPE-->LOC）<br/>
  用法 python change_tag.py input_file output_file<br/>
  param:<br/>
    input_file:输入文件，待替换<br/>
    output_file:输出文件，已替换<br/>

util/ner_metrics.py<br/>
  功能：计算metrics,各类型的f1.<br/>
  用法 python ner_metrics.py input_file<br/>
  param:<br/>
    input_file：三列，第一列是文本字符，第二列是标注label，第三列是预测label<br/>

# general_ner_predict_sdk
  推理sdk
  ## 对齐问题
  bert的tokenizer会有split和join操作，tokenize之后的每个token很多情况下不能和原始文本对齐。
  bert对原始文本处理后，会有[UNK], 不能和原始文本对齐。
  bert的中文预训练模型要求对于字母to_lower，不容易和原始文本对齐。

    例子：
    原始文本：曹œ斌是机器学习和自然语言处理专家，香港  科技 大学博士。曾任职于微软研究院、Bing 搜索，担任 CorTana 首席算法科学家
    tokenize后文本：曹 [UNK] 斌 是 机 器 学 习 和 自 然 语 言 处 理 专 家 ， 香 港 科 技 大 学 博 士 。 曾 任 职 于 微 软 研 究 院 、 bing 搜 索 ， 担 任 co ##rt ##ana 首 席 算 法 科 学 家

    比如 “曹œ斌” 是PER，但是tokenize推理之后ids2token后难以和原文本对齐。

    解决方法：
    对于tokenize后的每个token，建立指向原文本的索引。 具体代码见general_ner_predict_sdk/sentence.py
    


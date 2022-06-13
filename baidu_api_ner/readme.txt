使用方法：
    1. 调用api计算ner:
        python baidu_api_ner.py ../data/ontonotes_data/data_test output/ontonotes_test.result.all_tag
    2. 为了计算metrics，删除api未定义的NER类型，替换NER类型，需要根据数据集和api修改代码中RESERVE_LIST和REPLACE_MAP变量：
        python ../util/change_tag.py output/ontonotes_test.result.all_tag output/ontonotes_test.result.sub_tag
    3. 计算metrics，需要根据api定义的实体类型修改代码中的T1变量
        python ../util/ner_metrics.py output/ontonotes_test.result.sub_tag.fix_tag

baidu_api_ner.py
    功能：调用baidu_api计算ner，输入输出均为标准NER格式:第一列为字符，如果有标注label第二列为标注label,第三列为预测label，如果没有标注label, 第二列为预测label. 能识别的NER类型包括：
        PER
        ORG
        LOC
        TIME
    用法：python baidu_api_ner.py input_file output_file
        param:
            input_file：输入的标准NER格式的文件，一列或两列。
            output_file：输出的标准NER格式的文件，两列或三列。

metrics:
    ontonotes_test数据集：
        类型替换：
            GPE-->LOC
            NORP-->LOC 
        the accuracy is: 0.931022
        the macro F1 is: 0.709029
        the micro F1 is: 0.717016
        ================================================================================
        tp = 14293      tn = 121329
        ================================================================================
        tag             tp      fp      fn      Precsion        Recall          F1
        B-PER           1007    135     304     0.881786        0.768116        0.821035
        I-PER           2223    309     741     0.877962        0.750000        0.808952
        B-ORG           665     247     544     0.729167        0.550041        0.627063
        I-ORG           2449    908     1717    0.729520        0.587854        0.651070
        B-LOC           2174    280     845     0.885901        0.720106        0.794445
        I-LOC           3365    627     1488    0.842936        0.693386        0.760882
        B-TIME          739     514     468     0.589785        0.612262        0.600813
        I-TIME          1671    637     1518    0.724003        0.523989        0.607968
        ================================================================================
    ChineseNer_test数据集：
        the accuracy is: 0.961811
        the macro F1 is: 0.793270
        the micro F1 is: 0.785644
        ================================================================================
        tp = 18388      tn = 192438
        ================================================================================
        tag             tp      fp      fn      Precsion        Recall          F1
        B-PER           1522    119     342     0.927483        0.816524        0.868474
        I-PER           3005    352     596     0.895144        0.834490        0.863754
        B-ORG           1352    374     833     0.783314        0.618764        0.691383
        I-ORG           5920    1198    2836    0.831694        0.676108        0.745874
        B-LOC           2666    410     992     0.866710        0.728814        0.791803
        I-LOC           3923    957     1025    0.803893        0.792846        0.798331
        ================================================================================

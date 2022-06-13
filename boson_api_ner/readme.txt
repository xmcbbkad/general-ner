使用方法：
    1. 调用api计算ner: 
        python boson_api_ner.py ../data/ontonotes_data/data_test output/ontonotes_test.result.all_tag
    2. 为了计算metrics，删除api未定义的NER类型，替换NER类型，需要根据数据集和api修改代码中RESERVE_LIST和REPLACE_MAP变量：
        python ../util/change_tag.py output/ontonotes_test.result.all_tag output/ontonotes_test.result.sub_tag
    3. 调用api是满4000字符发起一次调用，句子以'\n'分割，但是boson的api会自动删除'\n',所以用特殊字符替代，会导致api把识别为实体开头，人工修复此问题:
        python fix_tag.py output/ontonotes_test.result.sub_tag output/ontonotes_test.result.sub_tag.fix_tag
    4. 计算metrics，需要根据api定义的实体类型修改代码中的T1变量
        python ../util/ner_metrics.py output/ontonotes_test.result.sub_tag.fix_tag

boson_api_ner.py
    功能：调用boson_api计算ner，输入输出均为标准NER格式:第一列为字符，如果有标注label第二列为标注label,第三列为预测label，如果没有标注label，第二列为预测label. 能识别的NER类型包括：
        time-->TIME
        location-->LOC
        person_name-->PER
        org_name-->ORG
        company_name-->ORG
        product_name-->DEV
        job_title-->目前不用
    用法：python boson_api_ner.py input_file output_file
        param:
            input_file：输入的标准NER格式的文件，一列或两列。
            output_file：输出的标准NER格式的文件，两列或三列。

metrics:
    ontonotes_test数据集:
        类型替换：
            GPE-->LOC
            NORP-->LOC
        the accuracy is: 0.942191
        the macro F1 is: 0.631316
        the micro F1 is: 0.754427
        ================================================================================
        tp = 16018      tn = 121231
        ================================================================================
        tag             tp      fp      fn      Precsion        Recall          F1
        B-PER           1124    196     187     0.851515        0.857361        0.854428
        I-PER           2571    431     393     0.856429        0.867409        0.861884
        B-ORG           864     528     345     0.620690        0.714640        0.664360
        I-ORG           3206    1454    960     0.687983        0.769563        0.726490
        B-LOC           1833    421     803     0.813221        0.695372        0.749693
        I-LOC           2963    935     1197    0.760133        0.712260        0.735418
        B-TIME          902     134     305     0.870656        0.747307        0.804280
        I-TIME          2541    238     648     0.914358        0.796802        0.851542
        B-DEV           5       182     15      0.026738        0.250000        0.048309
        I-DEV           9       1016    40      0.008780        0.183673        0.016760
        ================================================================================

    ChineseNer_test数据集：
        the accuracy is: 0.968941
        the macro F1 is: 0.819932
        the micro F1 is: 0.815912
        ================================================================================
        tp = 19783      tn = 192606
        ================================================================================
        tag             tp      fp      fn      Precsion        Recall          F1
        B-PER           1660    121     204     0.932061        0.890558        0.910837
        I-PER           3306    267     295     0.925273        0.918078        0.921662
        B-ORG           1599    684     586     0.700394        0.731808        0.715756
        I-ORG           6988    1483    1768    0.824932        0.798081        0.811285
        B-LOC           2475    203     1183    0.924197        0.676599        0.781250
        I-LOC           3755    940     1193    0.799787        0.758892        0.778803
        ================================================================================

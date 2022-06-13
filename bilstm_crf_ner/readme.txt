使用方法：
    python ner_bisltm_crf.py, 修改代码中的DO_TRAIN,DO_EVAL,DO_PREDICT变量控制训练/评估/预测, 修改TRAIN_FILE,DEV_FILE,TEST_FILE指定数据集.

metrics:
    ontonotes_test数据集：
        the accuracy is: 0.936501
        the macro F1 is: 0.638412
        the micro F1 is: 0.770334
        ================================================================================
        tp = 20737      tn = 114033
        ================================================================================
        tag             tp      fp      fn      Precsion        Recall          F1
        B-PER           1113    165     183     0.870892        0.858796        0.864802
        I-PER           2543    330     376     0.885137        0.871189        0.878108
        B-ORG           730     245     473     0.748718        0.606816        0.670340
        I-ORG           2956    897     1194    0.767194        0.712289        0.738723
        B-LOC           198     119     337     0.624606        0.370093        0.464789
        I-LOC           578     396     678     0.593429        0.460191        0.518386
        B-TIME          1042    190     151     0.845779        0.873428        0.859381
        I-TIME          2873    349     291     0.891682        0.908028        0.899781
        B-NORP          159     36      219     0.815385        0.420635        0.554974
        I-NORP          334     77      348     0.812652        0.489736        0.611162
        B-GPE           1654    639     418     0.721326        0.798263        0.757847
        I-GPE           2390    925     486     0.720965        0.831015        0.772089
        B-DEV           0       1       20      0.000000        0.000000        0.000000
        I-DEV           0       29      49      0.000000        0.000000        0.000000
        B-PERCENT       130     13      17      0.909091        0.884354        0.896552
        I-PERCENT       664     19      42      0.972182        0.940510        0.956084
        B-CUR           166     13      31      0.927374        0.842640        0.882979
        I-CUR           940     42      50      0.957230        0.949495        0.953347
        B-QUANTITY      48      18      24      0.727273        0.666667        0.695652
        I-QUANTITY      202     65      57      0.756554        0.779923        0.768061
        B-CARDINAL      501     180     316     0.735683        0.613219        0.668892
        I-CARDINAL      619     215     138     0.742206        0.817701        0.778127
        B-EVENT         71      37      134     0.657407        0.346341        0.453674
        I-EVENT         301     149     500     0.668889        0.375780        0.481215
        B-WORK          43      22      49      0.661538        0.467391        0.547771
        I-WORK          142     90      191     0.612069        0.426426        0.502655
        B-LAN           11      2       14      0.846154        0.440000        0.578947
        I-LAN           19      4       23      0.826087        0.452381        0.584615
        B-LAW           1       2       35      0.333333        0.027778        0.051282
        I-LAW           37      21      147     0.637931        0.201087        0.305785
        B-ORDINAL       134     19      18      0.875817        0.881579        0.878689
        I-ORDINAL       138     20      27      0.873418        0.836364        0.854489
        ================================================================================


训练记录：
1. 2_Layer_BiLSTM_base
  1.1 结构和参数：
    sentence_length=150, word_dim=200
    两层BiLSTM，第一层units=128，第二层units=64，dropout均为0.5
    batch_size=500, epochs=20
  1.2 结果：
    ontonotes valid dataset:    loss: 0.048 acc: 0.985  macro F1: 0.584 micro F1: 0.747
    ontonotes test dataset:     loss: 0.049 acc: 0.985  macro F1: 0.574 micro F1: 0.741
    wiki valid dataset: loss: 0.024 acc: 0.990  macro F1: 0.847 micro F1: 0.817
    wiki test dataset:  loss: 0.024 acc: 0.990   macro F1: 0.855    micro F1: 0.818
  1.3 结论：
    1.3.1 去掉一层BiLSTM后macroF1升了，microF1降了，不明显，一层还是两层bi-lstm没什么影响。
--------------------------------------------------------------------------------
2. 2_Layer_BiLSTM_CRF_base
  2.1 结构和参数：
    和1相比，加入了CRF。
  2.2 结果：
    ontonotes valid dataset:    loss: 0.035 acc: 0.986  macro F1: 0.606 micro F1: 0.752
    ontonotes test dataset:     loss: 0.035 acc: 0.985  macro F1: 0.595 micro F1: 0.744
    wiki valid dataset: loss: -0.187    acc: 0.990  macro F1: 0.853 micro F1: 0.818
    wiki test dataset:  loss: -0.187    acc: 0.990  macro F1: 0.857 micro F1: 0.818
  2.3 结论：
    2.3.1 加了CRF用处不大。
--------------------------------------------------------------------------------
3. 2_Layer_BiLSTM_CRF_remove_LSTM_dropout
  3.1 结构和参数：
    和2相比，去掉了LSTM的dropout.
  3.2 结果：
    ontonotes valid dataset:    loss: 0.037 acc: 0.987  macro F1: 0.684 micro F1: 0.778
    ontonotes test dataset:     loss: 0.036 acc: 0.987  macro F1: 0.675 micro F1: 0.772
  3.3 结论：
    去掉dropout后效果反而好，LSTM的dropout=0.5估计太大了。把dropout调成0.2，效果会好一点(如下)，说明dropout有点用。
      ontonotes valid dataset:    loss: 0.031 acc: 0.988  macro F1: 0.685 micro F1: 0.789
      ontonotes test dataset:     loss: 0.031 acc: 0.987  macro F1: 0.675 micro F1: 0.777
    把LSTM中的dropout去掉，把Dropout=0.2单独接在BiLSTM之后，会好一点点(如下)：
      ontonotes valid dataset:    loss: 0.034 acc: 0.988  macro F1: 0.677 micro F1: 0.785
      ontonotes test dataset:     loss: 0.034 acc: 0.987  macro F1: 0.677 micro F1: 0.778
4. 2_Layer_BiLSTM_TimeDistributed_CRF
  4.1 结构和参数：
    和3相比，在BiLSTM和CRF之间增加了TimeDistributed层，相当于送入CRF之前，把每个token由128维降到29维
  4.2 结果：
    ontonotes valid dataset:    loss: 0.039 acc: 0.987  macro F1: 0.686 micro F1: 0.781
    ontonotes test dataset:     loss: 0.040 acc: 0.987  macro F1: 0.682 micro F1: 0.775
  4.3 结论：
    发现没什么用。重复实验，发现结果偏差较大，应该是还没收敛，epoch=20小了，应该测一下多少epoch会过拟合。试了一下发现epoch=19最好，之后略有下降，差别不大。

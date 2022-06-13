import numpy
import keras
from keras.models import Model
from collections import OrderedDict

T1 = ['O','B-PER', 'I-PER', 'B-TIME', 'I-TIME', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-NORP', 'I-NORP', 'B-GPE', 'I-GPE', 'B-DEV', 'I-DEV',  'B-PERCENT', 'I-PERCENT', 'B-CUR', 'I-CUR', 'B-QUANTITY', 'I-QUANTITY', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK', 'I-WORK', 'B-LAN', 'I-LAN', 'B-LAW', 'I-LAW', 'B-ORDINAL', 'I-ORDINAL']

T2 = OrderedDict()
for index,item in enumerate(T1):
    T2[item] = index


def my_F1(y_true, y_pred):
    
    print('=== begin my_F1 ===')
    # print(y_true.shape)
    # print(y_pred.shape)
    
    if y_true.shape != y_pred.shape:
        print('=== Prediction & Result Dismatch ===')
        return 0, 0, 0, 0, 0, 0
    
    typenum = len(y_true[0][0])
    
    t = numpy.zeros(typenum)
    fp = numpy.zeros(typenum)
    fn = numpy.zeros(typenum)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            tag_true = numpy.where(y_true[i][j] == numpy.max(y_true[i][j]))[0][0]
            tag_pred = numpy.where(y_pred[i][j] == numpy.max(y_pred[i][j]))[0][0]
            
            if tag_true == tag_pred:
                t[tag_true] += 1
            elif tag_true == 0 and tag_pred != 0:
                fp[tag_pred] += 1
            elif tag_true != 0 and tag_pred == 0:
                fn[tag_true] += 1
            else:
                fn[tag_true] += 1
                fp[tag_pred] += 1
    
    delta = 1e-20
    accuracy = (sum(t) + delta) / (len(y_true) * len(y_true[0]) + delta)
    
    f1 = numpy.zeros(typenum)
    for i in range(typenum)[1:]:
        f1[i] = (2 * t[i] + delta) / (2 * t[i] + fp[i] + fn[i] + delta)
    macro_F1 = numpy.mean(f1[1:])
    
    micro_F1 = (2 * sum(t[1:]) + delta) / (2 * sum(t[1:]) + sum(fp[1:]) + sum(fn[1:]) + delta)
    
    return accuracy , macro_F1 , micro_F1 , t , fp, fn 

################################################################
# my_evaluate()函数用于评估测试的模型效果
################################################################
def my_evaluate(y_true, y_pred, verbose=True):
    accuracy , macro_F1 , micro_F1 , t , fp, fn = my_F1(y_true, y_pred)
    print('the accuracy is: %f' % accuracy)
    print('the macro F1 is: %f' % macro_F1)
    print('the micro F1 is: %f' % micro_F1)
    
    if not verbose:
        return 

    print('================================================================================')
    print('tp = %d\ttn = %d' % (sum(t[1:]),t[0]))
    print('================================================================================')
    
    print('tag\t\ttp\tfp\tfn\tPrecsion\tRecall\t\tF1')
    delta = 1e-20

    for k in T2:
        if k in ['B-PERCENT','I-PERCENT','B-QUANTITY','I-QUANTITY','B-CARDINAL','I-CARDINAL','B-ORDINAL','I-ORDINAL']:
            print('%s\t%d\t%d\t%d\t%f\t%f\t%f' % (k, t[T2[k]], fp[T2[k]], fn[T2[k]], (t[T2[k]]+delta)/(t[T2[k]]+fp[T2[k]]+delta),
            (t[T2[k]]+delta)/(t[T2[k]]+fn[T2[k]]+delta), (2*t[T2[k]]+delta)/(2*t[T2[k]]+fp[T2[k]]+fn[T2[k]]+delta)))
        else:
            print('%s\t\t%d\t%d\t%d\t%f\t%f\t%f' % (k, t[T2[k]], fp[T2[k]], fn[T2[k]], (t[T2[k]]+delta)/(t[T2[k]]+fp[T2[k]]+delta),
            (t[T2[k]]+delta)/(t[T2[k]]+fn[T2[k]]+delta), (2*t[T2[k]]+delta)/(2*t[T2[k]]+fp[T2[k]]+fn[T2[k]]+delta)))
    print('================================================================================')
    
    return
 

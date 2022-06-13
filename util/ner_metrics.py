import numpy
import sys
from collections import OrderedDict

T1 = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME', 'B-NORP', 'I-NORP', 'B-GPE', 'I-GPE', 'B-DEV', 'I-DEV',  'B-PERCENT', 'I-PERCENT', 'B-CUR', 'I-CUR', 'B-QUANTITY', 'I-QUANTITY', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK', 'I-WORK', 'B-LAN', 'I-LAN', 'B-LAW', 'I-LAW', 'B-ORDINAL', 'I-ORDINAL']
#T1 = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME', 'B-DEV', 'I-DEV']
#T1 = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
#T1 = ['O','B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME']

T2 = OrderedDict()
for index,item in enumerate(T1):
    T2[item] = index


def my_F1(y_true, y_pred, is_str=True):
    
    print('=== begin my_F1 ===')
    # print(y_true.shape)
    # print(y_pred.shape)
    
    if y_true.shape != y_pred.shape:
        print('=== Prediction & Result Dismatch ===')
        return 0, 0, 0, 0, 0, 0
    
    typenum = len(T1)

    t = numpy.zeros(typenum)
    fp = numpy.zeros(typenum)
    fn = numpy.zeros(typenum)
    for i in range(len(y_true)):
        tag_true = T2[y_true[i]] if is_str else int(y_true[i])
        tag_pred = T2[y_pred[i]] if is_str else int(y_pred[i])
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
    accuracy = (sum(t) + delta) / (len(y_true) + delta)
    
    f1 = numpy.zeros(typenum)
    for i in range(typenum)[1:]:
        f1[i] = (2 * t[i] + delta) / (2 * t[i] + fp[i] + fn[i] + delta)
    macro_F1 = numpy.mean(f1[1:])
    
    micro_F1 = (2 * sum(t[1:]) + delta) / (2 * sum(t[1:]) + sum(fp[1:]) + sum(fn[1:]) + delta)
    
    #return accuracy , macro_F1 , micro_F1 , t , fp, fn 
    
    print('the accuracy is: %f' % accuracy)
    print('the macro F1 is: %f' % macro_F1)
    print('the micro F1 is: %f' % micro_F1)
    print('================================================================================')
    print('tp = %d\ttn = %d' % (sum(t[1:]),t[0]))
    print('================================================================================')
    
    print('tag\t\ttp\tfp\tfn\tPrecsion\tRecall\t\tF1')
    delta = 1e-20
    for k in T2:
        if k == 'O':
            continue
        if k in ['B-PERCENT','I-PERCENT','B-QUANTITY','I-QUANTITY','B-CARDINAL','I-CARDINAL','B-ORDINAL','I-ORDINAL']:
            print('%s\t%d\t%d\t%d\t%f\t%f\t%f' % (k, t[T2[k]], fp[T2[k]], fn[T2[k]], (t[T2[k]]+delta)/(t[T2[k]]+fp[T2[k]]+delta), 
            (t[T2[k]]+delta)/(t[T2[k]]+fn[T2[k]]+delta), (2*t[T2[k]]+delta)/(2*t[T2[k]]+fp[T2[k]]+fn[T2[k]]+delta)))
        else:
            print('%s\t\t%d\t%d\t%d\t%f\t%f\t%f' % (k, t[T2[k]], fp[T2[k]], fn[T2[k]], (t[T2[k]]+delta)/(t[T2[k]]+fp[T2[k]]+delta), 
            (t[T2[k]]+delta)/(t[T2[k]]+fn[T2[k]]+delta), (2*t[T2[k]]+delta)/(2*t[T2[k]]+fp[T2[k]]+fn[T2[k]]+delta)))
    
    print('================================================================================')
    
if __name__=='__main__':
    f_metrics = open(sys.argv[1], "r")

    list_true = []
    list_pre = []

    for line in f_metrics:
        list_line = line.strip().split()
        if len(list_line) != 3:
            continue

        token = list_line[0]
        p_true = list_line[1]
        p_pre = list_line[2]

        list_true.append(p_true)
        list_pre.append(p_pre)

    array_true = numpy.array(list_true)
    array_pre = numpy.array(list_pre) 
    
    my_F1(array_true, array_pre)

import torch
import torch.nn as nn
'''
在这里假设数据的维度都为【batch，datalen，features】
'''

def DF(data,m=6,n=6):
    '''
    down frequencey
    '''
    return torch.mean(data.unfold(1,m,n),axis=-1)

def DF_STD(data,m=6,n=6):
    '''
    down frequencey
    '''
    return torch.std(data.unfold(1,m,n),axis=-1)

def DF_MAX(data,m=6,n=6):
    '''
    down frequencey
    '''
    max_value, max_index = torch.max(data.unfold(1,m,n),axis=-1)

    return max_value

def neut_class(X,cls):

    num_classes = int(torch.max(cls).item() + 1)
    batch_size, time_series, num_features = X.shape
    out = torch.zeros_like(X, dtype=X.dtype)

    for i in range(num_classes):
        class_mask = cls == i
        for feature in range(num_features):
            # feature_mask = class_mask.unsqueeze(1).repeat(1, time_series)
            if torch.any(class_mask):  
                out[class_mask,:, feature] = torch.mean(X[class_mask,:, feature])  
    return out
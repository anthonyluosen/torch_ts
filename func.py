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
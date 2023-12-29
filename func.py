import torch
import torch.nn as nn
'''
在这里假设数据的维度都为【batch，datalen，features】
'''
EPS=1e-8
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

# def neut_class(X,cls):

#     num_classes = int(torch.max(cls).item() + 1)
#     batch_size, time_series, num_features = X.shape
#     out = torch.zeros_like(X, dtype=X.dtype)

#     for i in range(num_classes):
#         class_mask = cls == i
#         for feature in range(num_features):
#             # feature_mask = class_mask.unsqueeze(1).repeat(1, time_series)
#             if torch.any(class_mask):  
#                 out[class_mask,:, feature] = torch.mean(X[class_mask,:, feature])  
#     return out

def mean_class(X,cls):
    num_classes = int(torch.max(cls).item() + 1)
    # batch_size, time_series, num_features = X.shape

    # 初始化输出张量
    out = torch.zeros_like(X, dtype=X.dtype)
    for i in range(num_classes):
        class_mask = (cls == i)
        # 检查是否有属于该类的实例
        # if class_mask.any():
        # 为每个特征计算均值
        class_mean = torch.mean(X[class_mask], dim=0)
        # 使用扩展的掩码将均值赋值给相应的类别位置
        out[class_mask] = class_mean
    return out
def zscore_torch(X, axis=0):
    # Copy the input tensor
    tmpx = X.clone()

    # Set the values at 'nvalid' indices to NaN
    # tmpx[nvalid] = torch.nan
    mean = torch.mean(tmpx, dim=axis, keepdim=True)
    std = torch.std(tmpx, dim=axis, keepdim=True)
    # Perform standardization
    zscored = (tmpx - mean) / (std + EPS)

    return zscored

def dummy_class(cls):
    num_classes = int(torch.max(cls).item() + 1)
    # print(cls.shape)
    # 初始化输出张量
    out = torch.zeros((cls.shape[0],cls.shape[0]), dtype=torch.int).to(cls.device)
    for i in range(num_classes):
        class_mask = (cls == i)
        out[class_mask] = -class_mask.int()
    return out+1

def idx2onehot(idx, n):
    # 确保idx中的最大值小于n
    assert torch.max(idx).item() < n

    # 如果idx是一维的，将其转换为二维的（每行一个元素）
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    # 创建一个全零的矩阵，大小为(idx的行数, n)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)

    # 使用scatter_方法填充onehot矩阵，把对应的位置设为1
    onehot.scatter_(1, idx, 1)
    
    return onehot

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval
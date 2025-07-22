import numpy as np
from scipy.special import softmax
from sympy.physics.vector import gradient


#单个cell的前项传播过程
#两个输入x_t以及前一个隐藏层的输出h_prev， cell中的参数parameters
def rnn_cell_forward(x_t, h_prev, parameters):
    #取出参数
    U =parameters['U']
    W = parameters['W']
    V = parameters['V']
    bh = parameters['bh']
    by = parameters['by']

    #根据公式计算
    #隐藏层输出计算
    h_next = np.tanh(np.dot(U, x_t) + np.dot(W, h_prev) + bh) #双曲正切函数，值域在(-1, 1)

    #计算cell的输出
    out_pred = softmax(np.dot(V, h_next) + by)

    #记录每一层的值，用于反向传播计算
    cache = (h_next, h_prev, x_t, parameters)

    return h_next, out_pred, cache

def rnn_forward(x, h0, parameters):
    '''
    x是输入序列，形状[m, 1 T],T代表序列长度
    h0代表初始状态输入，0
    parameters代表所有cell共享的参数， U,W,V,bh,by
    '''
    #获取序列长度，即时刻数
    m, _, T = x.shape

    #获取输入的N，定义隐藏层输出大小
    m, n = parameters['V'].shape #V的形状是y x h y指的是输出的维度,h指的是隐层层的神经元个数即维度,此处说明输出的维度为m,隐藏层的维度为n

    #获取和h0保存到h_next,以便进行前向传播
    h_next = h0

    #定义h, y保存所有cell的隐藏层的状态及输出
    h = np.zeros((n, 1, T))
    y = np.zeros((m, 1, T))

    caches = []
    #循环对每一个cell进行前向传播计算
    for t in range(T):
        # 循环对每一个cell进行前向传播计算
        h_next, out_pred, cache = rnn_cell_forward(x[:, :, t], h_next, parameters)
        #放入数组中
        h[:, :, t] = h_next
        y[:, :, t] = out_pred
        #放入所有的缓存到列表
        caches.append(cache)

    return h, y, caches


def rnn_cell_backward(dh_next, cache):
    #获取cache中的缓存值和参数
    (h_next, h_prev, x_t, parameters) = cache
    U = parameters['U']
    W = parameters['W']
    

    #根据公式进行反向传播计算
    #计算tanh的导数
    dtanh = (1 - h_next ** 2) *dh_next

    #计算U的梯度值
    dU = np.dot(dtanh, x_t.T)

    #计算W的梯度值
    dW = np.dot(dtanh, h_prev.T)

    #计算ba的梯度值
    #保持计算之后的维度不变
    dba = np.sum(dtanh, axis=1, keepdims=1) #axis=1表示列求和

    #计算x_t的导数
    dx_t = np.dot(U.T, dtanh)

    #计算h_prev的导数
    dh_prev = np.dot(W.T, dtanh)

    #把所有的导数保存到字典返回
    gradients = {'dtanh': dtanh, 'dU':dU, 'dW':dW, 'dba':dba, 'dx_t':dx_t, 'dh_prev':dh_prev}

    return gradients


def rnn_backward(dh, caches):
    """
    dh: 所有时间步隐藏状态的梯度，shape = [n, 1, T]
    caches: 每个时间步前向传播时保存的cache列表
    """
    (h_next, h_prev, x_t, parameters) = caches[0]
    U = parameters['U']
    W = parameters['W']

    n, m = h_next.shape  # 隐藏层维度 n，batch大小 m
    _, _, T = dh.shape
    input_dim = x_t.shape[0]

    # 初始化所有参数的梯度
    dU = np.zeros_like(U)
    dW = np.zeros_like(W)
    dba = np.zeros((n, 1))
    dx = np.zeros((input_dim, 1, T))
    dh_prev = np.zeros((n, 1))

    # 从后往前遍历时间步
    for t in reversed(range(T)):
        dh_t = dh[:, :, t] + dh_prev  # 当前步的梯度 + 来自下一个时间步的梯度
        gradients = rnn_cell_backward(dh_t, caches[t])

        # 累加梯度
        dU += gradients['dU']
        dW += gradients['dW']
        dba += gradients['dba']
        dx[:, :, t] = gradients['dx_t']
        dh_prev = gradients['dh_prev']  # 留给下一个时间步使用

    grads = {'dU': dU, 'dW': dW, 'dba': dba, 'dx': dx, 'dh0': dh_prev}
    return grads



if __name__ == '__main__':
    #rnn_cell_forward测试
    np.random.seed(1)
    #定义cell的输入
    x_t = np.random.randn(3, 1) #使用randn使输入满足标准正态分布
    h_prev = np.random.randn(5, 1)

    #定义参数
    U = np.random.randn(5, 3)
    W = np.random.randn(5, 5)
    V = np.random.randn(3, 5)
    bh = np.random.randn(5, 1)
    by = np.random.randn(3, 1)

    parameters = {'U': U, 'W': W, 'V': V, 'bh': bh, 'by': by}
    h_next, out_pred, cache = rnn_cell_forward(x_t, h_prev, parameters)
    print('h_next:', h_next)
    print('h_next.shape', h_next.shape)
    print('out_pred:', out_pred)
    print('out_pred.shape', out_pred.shape)

    #rnn_forward测试
    np.random.seed(1)
    # 定义4个cell，每个词的形状为[3, 1]
    x = np.random.randn(3, 1, 4)  # 使用randn使输入满足标准正态分布
    h0 = np.random.randn(5, 1)

    # 定义参数
    U = np.random.randn(5, 3)
    W = np.random.randn(5, 5)
    V = np.random.randn(3, 5)
    bh = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {'U': U, 'W': W, 'V': V, 'bh': bh, 'by': by}

    h, y, caches = rnn_forward(x, h0, parameters)
    print('h:', h)
    print('h.shape', h.shape)
    print('y:', y)
    print('y.shape', y.shape)

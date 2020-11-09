import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

def compute_class_weights(histogram):
    classWeights = np.ones(6, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(6):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights

def focal_loss_zhihu(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    number_5 = torch.sum(target == 5).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights = torch.from_numpy(classWeights).float()
    weights=weights[target.view(-1)]#这行代码非常重要

    gamma = 2

    P = F.softmax(inputs, dim=1)#shape [num_samples,num_classes]

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    log_p = probs.log()

    # print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

    # print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss


class Kaggle_FocalLoss(nn.Module):
    """
    可以通过设定alpha的值:控制正负样本的权重
        - 1这个类的样本数比-1这个类的样本数多很多，那么a会取0到0.5来增加-1这个类的样本的权重）来控制正负样本对总的loss的共享权重。
        - 这里当a=0.5时就和标准交叉熵一样了（系数是个常数）
    gamma:控制容易分类和难分类样本的权重
        - 当 γ=0的时候，focal loss就是传统的交叉熵损失，当 γ 增加的时候，调制系数也会增加
    γ=0是标准的交叉熵损失；当γ增加的时候，a需要减小一点
    """
    def __init__(self, alpha=0.2, gamma=2, logits=False, reduce=True):
        super(Kaggle_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == '__main__':
    # #     pred=torch.rand((2,6,5,5))
    # pred = torch.rand((1, 3, 320, 320))
    # #     y=torch.from_numpy(np.random.randint(0,6,(2,5,5)))
    # y = torch.from_numpy(np.random.randint(0, 2, (1, 320, 320)))
    # loss2 = focal_loss_zhihu(pred, y)
    # print('pred:', pred.shape)
    # print('y:', y.shape)
    # print('loss2：', loss2)

    criterion = Kaggle_FocalLoss()
    output = torch.rand((2, 3, 320, 320))
    y_train = torch.from_numpy(np.random.randint(0, 2, (2, 3, 320, 320)))
    loss = criterion(output, y_train)
    print('output:', output.shape)
    print('y_train:', y_train.shape)
    print('loss2：', loss)
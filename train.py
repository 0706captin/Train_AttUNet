# coding=utf-8


from eval import eval_net
from torch import optim
from models.attention_model import Unet_att
from utils.load import *
from utils.criterion import *
from utils.visualise import *
from utils.others import *
import time
from config import config
from utils.loss import focal_loss_zhihu, Kaggle_FocalLoss
from utils.others import *
import os


def train_valid(i, net, num_epochs, learning_rate, weight_decay, batch_size, date):
    dir_checkpoint = os.path.join(opt.save_path, date)
    mkfile(dir_checkpoint)
    dataset = Glaucoma_Dataset(opt.data_path_train, phase='train')
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(opt.validation_split * dataset_size))
    if opt.shuffle_dataset:
        np.random.seed(opt.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

    train_loader = DATA.DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=opt.num_workers)
    validation_loader = DATA.DataLoader(dataset,
                                        batch_size=1,
                                        sampler=valid_sampler,
                                        drop_last=True,
                                        num_workers=opt.num_workers)
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = Kaggle_FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=learning_rate,
    #                       momentum=0.9,
    #                       nesterov=True,
    #                       weight_decay=weight_decay)  # 这里有正则化，用于处理过拟合
    # learning rate decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)

    train_ls, valid_ls = [], []  # 存储train_loss,test_loss
    train_acc, train_iou, train_dice = [], [], []
    train_epoch_acc, train_epoch_iou, train_epoch_dice = [], [], []
    valid_epoch_acc, valid_epoch_iou, valid_epoch_dice = [], [], []
    preLoss = 999
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, num_epochs))
        net.train()
        train_epoch_loss = 0
        newLoss = 0
        for s, [X_train, y_train, picture_name] in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # print('X_train:\n', X_train.shape)
            # print('y_train:\n', y_train.shape)
            # print(y_train.requires_grad)

            output = net(X_train)
            # print('output:\n', output.shape)
            loss = criterion(output, y_train)
            # loss = focal_loss_zhihu(output, y_train)
            # 得到每一个epoch的每一个step的loss叠加到train_epoch_loss里
            train_epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 算每个epoch的平均acc
            output1 = (output > 0.5).float()
            train_step_acc, train_step_iou, train_step_dice = log_rmse(y_train, output1)
            train_acc.append(train_step_acc)
            train_iou.append(train_step_iou)
            train_dice.append(train_step_dice)
        train_epoch_acc.append(np.mean(train_acc))
        train_epoch_iou.append(np.mean(train_iou))
        train_epoch_dice.append(np.mean(train_dice))

        # 算出一个epoch的loss并把这个epoch的loss放在train_ls里面
        # train_ls存的是每个epoch的loss
        newLoss = train_epoch_loss / len(train_loader)
        train_ls.append(newLoss)
        if preLoss < newLoss:
            opt.lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
        else:
            preLoss = newLoss
        print('lr:', opt.lr)
        print('Epoch finished ! Loss: {}'.format(newLoss))

        # validation   其中只有valid_hd是只有最后一个epoch的数值
        valid_acc, valid_iou, valid_dice, name = eval_net(i, net, num_epochs, epoch, validation_loader, date)
        # valid_ls.append(test_epoch_loss / lll)
        valid_epoch_acc.append(np.mean(valid_acc))
        valid_epoch_iou.append(np.mean(valid_iou))
        valid_epoch_dice.append(np.mean(valid_dice))

        print('train_acc:{} || test_acc:{}'.format(np.mean(train_acc), np.mean(valid_acc)))

        # 保存网络
        a = epoch + 1
        # print('a=', a)
        if a % opt.save_epoch == 0:
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, 'CP{}_k{}.pth'.format(epoch + 1, i + 1)))
            print('Checkpoint {} saved !'.format(epoch + 1))

        scheduler.step()
    print("END epoch{}!".format(epoch + 1))
    total = [train_ls, valid_ls, train_epoch_acc, train_epoch_iou, train_epoch_dice, valid_epoch_acc, valid_epoch_iou,
             valid_epoch_dice]
    return total, name


def k_fold(k=10,
           num_epochs=320,
           learning_rate=0.001,
           weight_decay=0.0005,
           batch_size=40,
           date='today'):
    train_acc, valid_acc, train_iou, valid_iou, train_dice, valid_dice = [], [], [], [], [], []
    
    for i in range(k):

        net = Unet_att(3,opt.num_classes)

        net = net.to(device)
        total, name = train_valid(i, net, num_epochs, learning_rate, weight_decay, batch_size, date)

        # 存十次实验的求和平均：train_ls的最后一个数是第i次实验结束后的最后的loss（epoch=num_epochs时）
        # 放入最后一个epoch是值
        train_acc.append(total[2][-1])
        valid_acc.append(total[5][-1])
        train_iou.append(total[3][-1])
        valid_iou.append(total[6][-1])
        train_dice.append(total[4][-1])
        valid_dice.append(total[7][-1])

        # print('输出total长度：', len(total))
        print('输出train每个epoch的loss值:')
        print('train_loss_epoch:{}'.format(total[0]))
        print('输出000valid每个epoch的loss值:')
        print('valid_loss_epoch:{}'.format(total[1]))
        print('输出每个epoch的acc值:')
        print('train_acc_epoch:{}\n valid_acc_epoch:{}'.format(total[2], total[5]))
        print('输出最后一个epoch的值:')
        print('train_acc:{} \n valid_acc:{}\n\n'.format(total[2][-1], total[5][-1]))
        print('train_iou:{} \n valid_iou:{}\n\n'.format(total[3][-1], total[6][-1]))
        print('train_dice:{} \n valid_dice:{}\n\n'.format(total[4][-1], total[7][-1]))

        # 展示结果（train的loss，train和test的acc）
        # display_loss(total[0], total[1], i + 1)
        display_train_loss(total[0], i + 1, date)
        display_acc(total[2], total[5], i + 1, date)
        display_iou(total[3], total[6], i + 1, date)
        display_dice(total[4], total[7], i + 1, date)
        # k_fold_acc(total[2], total[5])

        break

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    # 体现步骤四
    print('train_acc_mean:%.4f\n' % (np.mean(train_acc)),
          'valid_acc_mean:%.4f\n' % (np.mean(valid_acc)),
          'train_iou_mean:%.4f\n' % (np.mean(train_iou)),
          'valid_iou_mean:%.4f\n' % (np.mean(valid_iou)),
          'train_dice_mean:%.4f\n' % (np.mean(train_dice)),
          'valid_dice_mean:%.4f\n' % (np.mean(valid_dice)))
    print('train_acc_sd:%.4f' % get_rmse(train_acc),
          'valid_acc_sd:%.4f' % get_rmse(valid_acc), )
    print('train_iou_sd:%.4f' % get_rmse(train_iou),
          'valid_iou_sd:%.4f' % get_rmse(valid_iou), )
    print('train_dice_sd:%.4f' % get_rmse(train_dice),
          'valid_dice_sd:%.4f' % get_rmse(valid_dice), )


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    opt = config()
    start_time = time.time()
    print('start_time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    # k-fold cross validation
    k_fold(k=opt.k,
           num_epochs=opt.num_epoch,
           learning_rate=opt.lr,
           weight_decay=opt.weight_decay,
           batch_size=opt.batch_size,
           date=opt.date)
    end_time = time.time()
    print('end_time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('total_time:', time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(end_time - start_time)))

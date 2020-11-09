from utils.criterion import *
from utils.visualise import *
from config import *
from PIL import Image

opt = config()

"""是 batchsize = 3的时候用"""
# def eval_net(i, net, num_epoch, epoch, valid_iter, date):
#
#     net.eval()
#     valid_acc, valid_iou, valid_dice = [], [], []
#     test_epoch_loss = 0
#     for step, [X, y, picture_name] in enumerate(valid_iter):
#         # 分批测试，每次一张，每次的loss加到test_epoch_loss里面，会加len(test_iter)这么多次
#         with torch.no_grad():
#             X_test = X.cuda()
#             y_test = y.cuda()
#             output = net(X_test)
#             # 二值化
#             output = (output > 0.5).float()
#             # 一步有一个acc，有len(test_iter)步，就有len(test_iter)个acc
#             test_step_acc, test_step_iou, test_step_dice = log_rmse(y_test, output)
#             valid_acc.append(test_step_acc)
#             valid_iou.append(test_step_iou)
#             valid_dice.append(test_step_dice)
#             if epoch == num_epoch - 1:
#                 # print("epoch{} save 3 output!".format(num_epoch))
#                 save3PIL(step, i, X_test, output, y_test, date, picture_name)
#     return valid_acc, valid_iou, valid_dice


'''是 batchsize = 1的时候用'''

def eval_net(i, net, num_epoch, epoch, valid_iter, date):

    net.eval()
    valid_acc, valid_iou, valid_dice= [], [], []
    test_epoch_loss = 0
    name = []
    for step, [X, y, picture_name] in enumerate(valid_iter):
        # 分批测试，每次一张，每次的loss加到test_epoch_loss里面，会加len(test_iter)这么多次
        with torch.no_grad():
            X_val = X.cuda()
            y_val = y.cuda()
            output = net(X_val)
            # 二值化
            output = (output > 0.5).float()
            # 一步有一个acc，有len(test_iter)步，就有len(test_iter)个acc
            val_step_acc, val_step_iou, val_step_dice = log_rmse(y_val, output)

            valid_acc.append(val_step_acc)
            valid_iou.append(val_step_iou)
            valid_dice.append(val_step_dice)

            name.append(picture_name[0])
            if epoch == num_epoch - 1:
                # print("epoch{} save 3 output!".format(num_epoch))
                save_1_output(i, X_val, output, y_val, picture_name[0], date)
                # 只算最后一个epoch的hd

                if step+1 == len(valid_iter):  # len(valid_iter)=85
                    # save_hd(valid_hd, name)
                    # print('valid_hd_mean:%.4f\n' % (np.mean(valid_hd)),
                    #       'valid_hd_sd:%.4f' % (get_rmse(valid_hd)))
                    save_score(i+1, name, valid_iou, valid_acc, valid_dice, date)

    return valid_acc, valid_iou, valid_dice, name
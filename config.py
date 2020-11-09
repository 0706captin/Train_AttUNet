
class config(object):
    # k-fold cross validation
    k = 2
    # focal loss's parameter
    alpha = 0.5
    gamma = 2

    data_path_train = '/home/chenxj/PycharmProjects/1_dataset'
    # data_path_test = '../dataset/eye/images_cropped_divide_index_5cls/test'
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    num_workers = 1
    lr = 0.0001

    lr_decay = 0.95
    weight_decay = 0.0
    use_gpu = True
    # model_path = "checkpoints/225epoch.pkl"
    # save file's name
    save_path = "./checkpoints"
    date = '1109'

    num_epoch = 21
    num_classes = 1
    batch_size = 16
    save_epoch = 20

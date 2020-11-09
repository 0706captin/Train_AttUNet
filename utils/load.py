import torch.utils.data as DATA
from PIL import Image
from utils.visualise import *
import torchvision.transforms.functional as tf
import random
# 0909
'''transform2 = transforms.Compose([
    transforms.Resize(480, interpolation=2),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform3 = transforms.Compose([
    transforms.Resize(480, interpolation=2),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
])
'''
transform2 = transforms.Compose([
    transforms.Resize((480, 480)),
    # transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform3 = transforms.Compose([
    transforms.Resize((480, 480)),
    # transforms.CenterCrop(480),
    transforms.ToTensor(),
])
# 0909


def transform1(image, mask):
    # 1) 50%的概率应用垂直，水平翻转。
    p1 = random.random()
    p2 = random.random()
    p3 = random.random()
    if p1 > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if p2 > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    if p3 > 0.5:
        # 2) 拿到角度的随机数。angle是一个-180到180之间的一个数
        angle = transforms.RandomRotation.get_params([30, 60])
        # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
        image = image.rotate(angle)
        mask = mask.rotate(angle)

    return image, mask


class Glaucoma_Dataset(DATA.Dataset):
    def __init__(self, root, phase):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试集划分数据
        '''
        super(Glaucoma_Dataset, self).__init__()

        self.root = root
        self.phase = phase

        self.dir_img = os.path.join(self.root, 'image_data')  # 训练图像文件
        self.dir_label = os.path.join(self.root, 'label_data')  # 图像的结果文件夹

        self.dir_img = os.path.join(self.dir_img, self.phase)  # 训练图像文件
        self.dir_label = os.path.join(self.dir_label, self.phase)  # 图像的结果文件夹

        # image
        self.imgs = os.listdir(self.dir_img)  # 输出的是[文件名]，包含.jpg——用于训练的原图
        self.img = [os.path.join(self.dir_img, s) for s in self.imgs]  # `os.path.join`是将两个路径名字粘贴在一起
        # label
        self.label = [os.path.join(self.dir_label, ss) for ss in self.imgs]

        self.name = self.imgs

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img, label, name = self.img[index], self.label[index], self.name[index]  # 此时的img，label只是对应图片的路径，还没有打开文件
        # 打开图片
        img = Image.open(img)
        label = Image.open(label)

        if self.phase == 'train':
            # img_1, label_1 = transform1(img, label)
            img_1, label_1 = img, label
            img = transform2(img_1)
            label = transform3(label_1)
        else:
            img = transform2(img)
            label = transform3(label)

        return img, label, name

    def __len__(self):
        '''
        返回数据集中所有的的图片个数
        '''
        return len(self.img)


if __name__ == '__main__':
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v6.1/segmentation_data'

    dataset = Glaucoma_Dataset(data_path_train, 'train')
    batch_size = 5
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

    train_loader = DATA.DataLoader(dataset, batch_size=batch_size,
                                   sampler=train_sampler)
    validation_loader = DATA.DataLoader(dataset, batch_size=batch_size,
                                        sampler=valid_sampler)

    # Usage Example:
    num_epochs = 1
    for epoch in range(num_epochs):
        # Train:
        for batch_index, (imgs, labels, name) in enumerate(train_loader):
            print('batch_index:', batch_index)
            print('imgs.shape:', imgs.shape)
            print('labels.shape:', labels.shape)
            print('name:', name)  # ('114-LI1.JPG', '032-RT.JPG', '061-RI3.JPG', '020-LS.JPG', '067-RS.JPG')

            imgs = imgtensor2im(imgs)
            plt.imshow(imgs)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()

            labels = labeltensor_to_PIL(labels[0])
            plt.imshow(labels)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()

            break
        # Valid:
        for batch_index, (imgs, labels, name) in enumerate(validation_loader):
            print('batch_index:', batch_index)
            print('imgs.shape:', imgs.shape)
            print('labels.shape:', labels.shape)
            print('name:', name)  # ('114-LI1.JPG', '032-RT.JPG', '061-RI3.JPG', '020-LS.JPG', '067-RS.JPG')

            imgs = imgtensor2im(imgs)
            plt.imshow(imgs)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()

            labels = labeltensor_to_PIL(labels[0])
            plt.imshow(labels)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()
            break
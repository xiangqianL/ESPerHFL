"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import os

import torchvision
from tqdm import tqdm
from sklearn import metrics
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser
from torch.utils.data.dataset import Subset

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def gen_ran_sum(_sum, num_users):
    base = 100*np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100*num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users

def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iid_esize_split(dataset, args, kwargs, is_shuffle = True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def iid_nesize_split(dataset, args, kwargs, is_shuffle = True):
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace = False)
        #dict_users[i] = dict_users[i].astype(int)
        #dict_users[i] = set(dict_users[i])
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)

    return data_loaders

def niid_esize_split(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 4* args.num_clients
    # the number of images in one shard
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        labels = dataset.targets
        #t = 6000
    else:
        labels = dataset.targets
        #t = 1000
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    lx = [100, 160, 40, 60, 101, 161, 41, 61, 102, 162, 42, 62, 103, 163, 43, 63, 104, 164, 44, 64,
          80, 140, 180, 181, 81, 141, 182, 183, 82, 142, 184, 185, 83, 143, 186, 187, 84, 144, 188, 189,
          65, 105, 120, 20, 66, 106, 121, 21, 67, 107, 122, 22, 68, 108, 123, 23, 69, 109, 124, 24,
          145, 146, 110, 165, 147, 148, 111, 166, 149, 150, 112, 167, 151, 152, 113, 168, 153, 154, 114, 169,
          45, 190, 85, 125, 46, 191, 86, 126, 47, 192, 87, 127, 48, 193, 88, 128, 49, 194, 89, 129,
          0, 25, 90, 155, 1, 26, 91, 156, 2, 27, 92, 157, 3, 28, 93, 158, 4, 29, 94, 159,
          115, 5, 170, 95, 116, 6, 171, 96, 117, 7, 172, 97, 118, 8, 173, 98, 119, 9, 174, 99,
          50, 70, 10, 130, 51, 71, 11, 131, 52, 72, 12, 132, 53, 73, 13, 133, 54, 74, 14, 134,
          55, 30, 135, 175, 56, 31, 136, 176, 57, 32, 137, 177, 58, 33, 138, 178, 59, 34, 139, 179,
          15, 35, 75, 195, 16, 36, 76, 196, 17, 37, 77, 197, 18, 38, 78, 198, 19, 39, 79, 199]
    #lx = [0,20,40,60,1,21,41,61,2,22,42,62,3,23,43,63,4,24,44,64,5,25,45,65,6,26,46,66,7,27,47,67,8,28,48,68,9,29,49,69,50,70,80,100,51,61,81,101,52,62,82,102,53,63,83,103,54,64,84,104,55,65,85,105,56,66,86,106,57,67,87,107,58,68,88,108,59,69,89,109,90,110,120,140,91,111,121,141,92,112,122,142,93,113,123,143,94,114,124,144,95,115,125,145,96,116,126,146,97,117,127,147,98,118,128,148,99,119,129,149,130,150,160,180,131,151,161,181,132,152,162,182,133,153,163,183,134,154,164,184,135,155,165,185,136,156,166,186,137,157,167,187,138,158,168,188,139,159,168,189,170,190,10,30,171,191,11,31,172,192,12,32,173,193,13,33,174,194,14,34,175,195,15,35,176,196,16,36,177,197,17,37,178,198,18,38,179,199,19,39]
    # xx={}
    #
    # for i in range(10):
    #     xx[i]=idxs[t*i:t*(i+1)]
    #
    # for p in range(10):
    #     index_1 = np.random.choice(xx[0].shape[0], num_imgs, replace=False)
    #     data1 = xx[0][index_1]
    #     index_2 = np.arange(xx[0].shape[0])
    #     index_2 = np.delete(index_2, index_1)
    #     xx[0]=xx[0][index_2]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data1]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_3 = np.random.choice(xx[1].shape[0], num_imgs, replace=False)
    #     data2 = xx[1][index_3]
    #     index_4 = np.arange(xx[1].shape[0])
    #     index_4 = np.delete(index_4, index_3)
    #     xx[1] = xx[1][index_4]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data2]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_5 = np.random.choice(xx[2].shape[0], num_imgs, replace=False)
    #     data3 = xx[2][index_5]
    #     index_6 = np.arange(xx[2].shape[0])
    #     index_6 = np.delete(index_6, index_5)
    #     xx[2] = xx[2][index_6]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data3]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_7 = np.random.choice(xx[3].shape[0], num_imgs, replace=False)
    #     data4 = xx[3][index_7]
    #     index_8 = np.arange(xx[3].shape[0])
    #     index_8 = np.delete(index_8, index_7)
    #     xx[3] = xx[3][index_8]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data4]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    # for p in range(10,20):
    #     index_1 = np.random.choice(xx[2].shape[0], num_imgs, replace=False)
    #     data1 = xx[2][index_1]
    #     index_2 = np.arange(xx[2].shape[0])
    #     index_2 = np.delete(index_2, index_1)
    #     xx[2] = xx[2][index_2]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data1]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_3 = np.random.choice(xx[3].shape[0], num_imgs, replace=False)
    #     data2 = xx[3][index_3]
    #     index_4 = np.arange(xx[3].shape[0])
    #     index_4 = np.delete(index_4, index_3)
    #     xx[3] = xx[3][index_4]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data2]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_5 = np.random.choice(xx[4].shape[0], num_imgs, replace=False)
    #     data3 = xx[4][index_5]
    #     index_6 = np.arange(xx[4].shape[0])
    #     index_6 = np.delete(index_6, index_5)
    #     xx[4] = xx[4][index_6]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data3]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_7 = np.random.choice(xx[5].shape[0], num_imgs, replace=False)
    #     data4 = xx[5][index_7]
    #     index_8 = np.arange(xx[5].shape[0])
    #     index_8 = np.delete(index_8, index_7)
    #     xx[5] = xx[5][index_8]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data4]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    # for p in range(20,30):
    #     index_1 = np.random.choice(xx[4].shape[0], num_imgs, replace=False)
    #     data1 = xx[4][index_1]
    #     index_2 = np.arange(xx[4].shape[0])
    #     index_2 = np.delete(index_2, index_1)
    #     xx[4] = xx[4][index_2]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data1]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_3 = np.random.choice(xx[5].shape[0], num_imgs, replace=False)
    #     data2 = xx[5][index_3]
    #     index_4 = np.arange(xx[5].shape[0])
    #     index_4 = np.delete(index_4, index_3)
    #     xx[5] = xx[5][index_4]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data2]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_5 = np.random.choice(xx[6].shape[0], num_imgs, replace=False)
    #     data3 = xx[6][index_5]
    #     index_6 = np.arange(xx[6].shape[0])
    #     index_6 = np.delete(index_6, index_5)
    #     xx[6] = xx[6][index_6]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data3]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_7 = np.random.choice(xx[7].shape[0], num_imgs, replace=False)
    #     data4 = xx[7][index_7]
    #     index_8 = np.arange(xx[7].shape[0])
    #     index_8 = np.delete(index_8, index_7)
    #     xx[7] = xx[7][index_8]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data4]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    # for p in range(30,40):
    #     index_1 = np.random.choice(xx[6].shape[0], num_imgs, replace=False)
    #     data1 = xx[6][index_1]
    #     index_2 = np.arange(xx[6].shape[0])
    #     index_2 = np.delete(index_2, index_1)
    #     xx[6] = xx[6][index_2]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data1]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_3 = np.random.choice(xx[7].shape[0], num_imgs, replace=False)
    #     data2 = xx[7][index_3]
    #     index_4 = np.arange(xx[7].shape[0])
    #     index_4 = np.delete(index_4, index_3)
    #     xx[7] = xx[7][index_4]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data2]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_5 = np.random.choice(xx[8].shape[0], num_imgs, replace=False)
    #     data3 = xx[8][index_5]
    #     index_6 = np.arange(xx[8].shape[0])
    #     index_6 = np.delete(index_6, index_5)
    #     xx[8] = xx[8][index_6]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data3]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_7 = np.random.choice(xx[9].shape[0], num_imgs, replace=False)
    #     data4 = xx[9][index_7]
    #     index_8 = np.arange(xx[9].shape[0])
    #     index_8 = np.delete(index_8, index_7)
    #     xx[9] = xx[9][index_8]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data4]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    # for p in range(40,50):
    #     index_1 = np.random.choice(xx[8].shape[0], num_imgs, replace=False)
    #     data1 = xx[8][index_1]
    #     index_2 = np.arange(xx[8].shape[0])
    #     index_2 = np.delete(index_2, index_1)
    #     xx[8] = xx[8][index_2]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data1]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_3 = np.random.choice(xx[9].shape[0], num_imgs, replace=False)
    #     data2 = xx[9][index_3]
    #     index_4 = np.arange(xx[9].shape[0])
    #     index_4 = np.delete(index_4, index_3)
    #     xx[9] = xx[9][index_4]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data2]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_5 = np.random.choice(xx[0].shape[0], num_imgs, replace=False)
    #     data3 = xx[0][index_5]
    #     index_6 = np.arange(xx[0].shape[0])
    #     index_6 = np.delete(index_6, index_5)
    #     xx[0] = xx[0][index_6]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data3]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    #     index_7 = np.random.choice(xx[1].shape[0], num_imgs, replace=False)
    #     data4 = xx[1][index_7]
    #     index_8 = np.arange(xx[1].shape[0])
    #     index_8 = np.delete(index_8, index_7)
    #     xx[1] = xx[1][index_8]
    #     dict_users[p] = np.concatenate((dict_users[p], idxs[data4]), axis=0)
    #     dict_users[p] = dict_users[p].astype(int)
    #
    # for i in range(50):
    #     data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
    #                                  batch_size=args.batch_size,
    #                                  shuffle=is_shuffle, **kwargs)


    #lx = [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19, 20, 30, 21, 31, 22, 32, 23, 33, 24, 34,
      #    25, 35, 26, 36, 27, 37, 28, 38, 29, 39, 40, 50, 41, 51, 42, 52, 43, 53, 44, 54, 45, 55, 46, 56, 47, 57, 48,
      #    58, 49, 59, 60, 70, 61, 71, 62, 72, 63, 73, 64, 74, 65, 75, 66, 76, 67, 77, 68, 78, 69, 79, 80, 90, 81, 91,
       #   82, 92, 83, 93, 84, 94, 85, 95, 86, 96, 87, 97, 88, 98, 89, 99]
    for i in range(args.num_clients):
        rand_set = {lx[4*i],lx[4*i+1],lx[4*i+2],lx[4*i+3]}

        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * (num_imgs): (rand + 1) * (num_imgs)]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)
    return data_loaders

def niid_esize_split_train(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
#     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
#     divide and assign
#     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test(dataset, args, kwargs, split_pattern,  is_shuffle = False ):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0]* args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace= False)
        # split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
            # store the label
            split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern

def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle = False ):
    """
    :param dataset: test dataset
    :param args:
    :param kwargs:
    :param split_pattern: split pattern from trainloaders
    :param test_size: length of testloader of each client
    :param is_shuffle: False for testloader
    :return:
    """
    data_loaders = [0] * args.num_clients
    # for mnist and cifar 10, only 10 classes
    num_shards = 10
    num_imgs = int (len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(len(dataset))
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
#     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i]
        # idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle = True):
    data_loaders = [0] * args.num_clients
    #one class perclients
    #any requirements on the number of clients?
    num_shards = args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 1, replace = False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand+1)*num_imgs]), axis = 0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                            batch_size = args.batch_size,
                            shuffle = is_shuffle, **kwargs)
    return data_loaders

def split_data(dataset, args, kwargs, is_shuffle = True):
    """
    return dataloaders
    """
    if args.iid == 1:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 0:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -1:
        data_loaders = iid_nesize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == -2:
        data_loaders = niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle)
    else :
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders

def get_dataset(dataset_root, dataset, args):
    trains, tests, test_loaders = {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders,  v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        raise ValueError('CODING ERROR: FEMNIST dataset should not use this file')
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders,  v_test_loader

# def data_tf(x):
#  x = np.array(x, dtype='float32') / 255
#  x = (x - 0.5) / 0.5 # 数据预处理，标准化
#  x = x.reshape((-1,)) # 拉平
#  x = torch.from_numpy(x)
#  return x

def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 6, 'pin_memory': True} if is_cuda else {}
    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = True,
                            download = True, transform = transform)
    test =  datasets.MNIST(os.path.join(dataset_root, 'mnist'), train = False,
                            download = True, transform = transform)
    #note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    #the actual batch_size may need to change.... Depend on the actual gradient...
    #originally written to get the gradient of the whole dataset
    #but now it seems to be able to improve speed of getting accuracy of virtual sequence
    # v_train_loader = DataLoader(train, batch_size = args.batch_size * args.num_clients,
    #                             shuffle = True, **kwargs)
    # v_test_loader = DataLoader(test, batch_size = args.batch_size * args.num_clients,
    #                             shuffle = False, **kwargs)
    v_test_loader=[]
    #for c in range(5):
    v_test_loader= test_loaders
    return  train_loaders, test_loaders, v_test_loader


def get_cifar10(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory':True} if is_cuda else{}
    if args.model == 'cnn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.model == 'resnet18':
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding = 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("this nn for cifar10 not implemented")
    # indices, indices_test = torch.load('./filter_r60_s01_n50_c.pt')
    # dataset_train = torchvision.datasets.CIFAR10(root='data', train=True, download=True,
    #                                              transform = transform_train)
    # dataset_test = torchvision.datasets.CIFAR10(root='data', train=False, download=True,
    #                                             transform = transform_test)
    # subsets = [Subset(dataset_train, indices[i]) for i in range(50)]
    # subsetst = [Subset(dataset_test, indices_test[i]) for i in range(50)]
    # train_loaders = [DataLoader(subset, batch_size=20, shuffle=True, num_workers=0) for subset in subsets]
    # test_loaders = [DataLoader(subsett, batch_size=20, shuffle=True, num_workers=0) for subsett in subsetst]
    # v_test_loader = test_loaders
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train = True,
                        download = True, transform = transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root,'cifar10'), train = False,
                        download = True, transform = transform_test)
    # v_train_loader = DataLoader(train, batch_size = args.batch_size,
    #                             shuffle = True, **kwargs)
    # v_test_loader = DataLoader(test, batch_size = args.batch_size,
    #                             shuffle = False, **kwargs)
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test,  args, kwargs, is_shuffle = False)
    v_test_loader = []
    # for c in range(5):
    v_test_loader = test_loaders
    return  train_loaders, test_loaders, v_test_loader

def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.targets
        # labels = dataloader.dataset.dataset.train_labels
    elif args.dataset == 'fsdd':
        labels = dataloader.dataset.labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)
    # print(num_samples)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution

if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)


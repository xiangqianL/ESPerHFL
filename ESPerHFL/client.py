# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
#from models.initialize_model import initialize_model
import copy
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

#from hierfavg import alpha_update
from models import *
import copy
import numpy as np

from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_cnn import mnist_lenet, mnist_lenet1
from models.mnist_logistic import LogisticRegression
from options import args_parser

# class Client():
#
#     def __init__(self, id, train_loader, test_loader, args, device):
#         self.id = id
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.model = initialize_model(args, device)
#         # copy.deepcopy(self.model.shared_layers.state_dict())
#         self.receiver_buffer = {}
#         self.batch_size = args.batch_size
#         #record local update epoch
#         self.epoch = 0
#         # record the time
#         self.clock = []
#
#     def local_update(self, num_iter, device):
#         itered_num = 0
#         loss = 0.0
#         end = False
#         # the upperbound selected in the following is because it is expected that one local update will never reach 1000
#         for epoch in range(1000):
#             for data in self.train_loader:
#                 inputs, labels = data
#                 inputs = Variable(inputs).to(device)
#                 labels = Variable(labels).to(device)
#                 loss += self.model.optimize_model(input_batch=inputs,
#                                                   label_batch=labels)
#                 itered_num += 1
#                 if itered_num >= num_iter:
#                     end = True
#                     # print(f"Iterer number {itered_num}")
#                     self.epoch += 1
#                     self.model.exp_lr_sheduler(epoch=self.epoch)
#                     # self.model.print_current_lr()
#                     break
#             if end: break
#             self.epoch += 1
#             self.model.exp_lr_sheduler(epoch = self.epoch)
#             # self.model.print_current_lr()
#         # print(itered_num)
#         # print(f'The {self.epoch}')
#         loss /= num_iter
#         return loss
#
#     def test_model(self, device):
#         correct = 0.0
#         total = 0.0
#         with torch.no_grad():
#             for data in self.test_loader:
#                 inputs, labels = data
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = self.model.test_model(input_batch= inputs)
#                 _, predict = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predict == labels).sum().item()
#         return correct, total
#
#     def send_to_edgeserver(self, edgeserver):
#         edgeserver.receive_from_client(client_id= self.id,
#                                         cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
#                                         )
#         return None
#
#     def receive_from_edgeserver(self, shared_state_dict):
#         self.receiver_buffer = shared_state_dict
#         return None
#
#     def sync_with_edgeserver(self):
#         """
#         The global has already been stored in the buffer
#         :return: None
#         """
#         # self.model.shared_layers.load_state_dict(self.receiver_buffer)
#         self.model.update_model(self.receiver_buffer)
#         return None

class Client():

    def __init__(self, id, train_loader, test_loader, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer1 = {}
        self.receiver_buffer2 = {}
        self.m=0
        # self.shared_layers={}
        # self.local={}
        self.batch_size = args.batch_size
        # record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []
        self.lr_decay = args.lr_decay
        self.lr_decay_epoch = args.lr_decay_epoch
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer_w = optim.Adam(params=self.shared_layers.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
        # self.optimizer_v = optim.Adam(params=self.local.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
        #                               weight_decay=0)
        self.optimizer_w = optim.SGD(params=self.shared_layers.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_v = optim.SGD(params=self.local.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.optimize_model(input_batch=inputs, label_batch=labels)
                # self.m = alpha_update(self.shared_layers, self.local,
                #                                        self.m)


                # param0 = self.shared_layers.state_dict()
                # param1 = self.local.state_dict()
                # param_new = {}
                # for key in param0.keys():
                #     if key=='gamma'or key=='beta':
                #         param_new[key]=param1[key]
                #     else :
                #         param_new[key]=param0[key]
                # self.shared_layers.load_state_dict(param_new)

                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            self.exp_lr_sheduler(epoch=self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # test
                self.shared_layers.train(False)
                with torch.no_grad():
                    #mixt=copy.deepcopy(self.local)
                    mixt=self.local
                    local_dict = self.local.state_dict()
                    share_dict =self.shared_layers.state_dict()
                    mixt_dict = mixt.state_dict()
                    for par1, par2,par3 in zip(mixt_dict,local_dict,share_dict):
                        mixt_dict[par1]=self.m *local_dict[par2]+(1-self.m)*share_dict[par3]
                    mixt.load_state_dict(mixt_dict)


                    # for k in mixt.parameters():
                    #    mixt[k]=self.m * self.local[k]+(1-self.m)*self.shared_layers[k]
                    outputs = mixt(inputs)
                self.shared_layers.train(True)
                # test end
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id=self.id,
                                       cshared_state_dict=copy.deepcopy(self.shared_layers.state_dict()),
                                       v=copy.deepcopy(self.local.state_dict()),
                                       a=self.m)
        return None

    def receive_from_edgeserver(self, shared_state_dict,mix,l_model):
        self.m=mix
        self.receiver_buffer1 = shared_state_dict
        self.receiver_buffer2 = l_model

        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.update_model(self.receiver_buffer1,self.receiver_buffer2)

        return None

    def initialize_model(self, args, device):
        if args.global_model:
            if args.dataset == 'mnist':
                if args.model == 'lenet':
                    self.shared_layers = mnist_lenet1(input_channels=1, output_channels=10)
                    self.local = mnist_lenet1(input_channels=1, output_channels=10)
                elif args.model == 'logistic':
                    self.shared_layers = LogisticRegression(input_dim=1, output_dim=10)
                    self.local = LogisticRegression(input_dim=1, output_dim=10)
                else:
                    raise ValueError('Model not implemented for MNIST')
            elif args.dataset == 'cifar10':
                if args.model == 'cnn':
                    self.shared_layers= cifar_cnn_3conv(input_channels=3, output_channels=10)
                    self.local = cifar_cnn_3conv(input_channels=3, output_channels=10)
                elif args.model == 'resnet18':
                    self.shared_layers = ResNet18()
                    self.local =ResNet18()
                else:
                    raise ValueError('Model not implemented for CIFAR-10')



            # To be continue ......

            else:
                raise ValueError('The dataset is not implemented yet')
            if args.cuda:
                self.shared_layers = self.shared_layers.cuda(device)
                self.local=self.local.cuda(device)
        else:
            raise ValueError('Wrong input for the --global_model, only one is valid')

    def optimize_model(self, input_batch, label_batch):
        # w models
        self.shared_layers.train(True)
        output_batch = self.shared_layers(input_batch)
        self.optimizer_w.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer_w.step()

        # V
        self.local.train(True)
        output1 = self.shared_layers(input_batch)
        output2 =self.local(input_batch)
        output = self.m * output2 + (1 - self.m) * output1
        self.optimizer_v.zero_grad()
        batch_loss_mix = self.criterion(output, label_batch)
        batch_loss_mix.backward()
        self.optimizer_v.step()


        return batch_loss_mix.item()





    def exp_lr_sheduler(self, epoch):
        """"""

        if  (epoch + 1) % self.lr_decay_epoch:
            return None
        for param_group in self.optimizer_w.param_groups:
            # print(f'epoch{epoch}')
            param_group['lr'] *= self.lr_decay
            return None

    def update_model(self, new_shared_layers,lmodel):
        # args = args_parser()
        # if args.model == 'lenet':
        #     # algorithm 1
        #     grad_alpha = 0
        #     for _, param in enumerate(self.shared_layers.state_dict()):
        #         # temp1 = torch.multiply(self.mxiture, self.shared_layers.state_dict()[param])
        #         # temp2 = torch.multiply(1 - self.mxiture, new_shared_layers[param])
        #         # temp = temp1 + temp2
        #         dif = self.shared_layers.state_dict()[param].data-new_shared_layers[param].data
        #         grad = self.mxiture * self.shared_layers.state_dict()[param].grad.data + (1 - self.mxiture) * new_shared_layers[param].grad.data
        #         grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        #     grad_alpha += 0.02 * self.mxiture
        #     alpha_n = self.mxiture - 0.1 * grad_alpha
        #     alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)
        #     return alpha_n

                #self.mxiture.update({param:temp})
        # To be continue ......
        self.shared_layers.load_state_dict(new_shared_layers)
        self.local.load_state_dict(lmodel)
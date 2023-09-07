import copy
import torch
from torch import nn
import math

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if k != 'bn1.num_batches_tracked' and k != 'bn2.num_batches_tracked':
        #if k !='bn1.num_batches_tracked'and k !='layer1.0.bn1.num_batches_tracked' and k !='layer1.0.bn2.num_batches_tracked' and k !='layer1.1.bn1.num_batches_tracked' and k !='layer1.1.bn2.num_batches_tracked' and k !='layer2.0.bn1.num_batches_tracked' and k !='layer2.0.bn2.num_batches_tracked' and k !='layer2.0.shortcut.1.num_batches_tracked' and k !='layer2.1.bn1.num_batches_tracked' and k !='layer2.1.bn2.num_batches_tracked'and k !='layer3.0.bn1.num_batches_tracked'and k !='layer3.0.bn2.num_batches_tracked'and k !='layer3.0.shortcut.1.num_batches_tracked'and k !='layer3.1.bn1.num_batches_tracked'and k !='layer3.1.bn2.num_batches_tracked'and k !='layer4.0.bn1.num_batches_tracked'and k !='layer4.0.bn2.num_batches_tracked'and k !='layer4.0.shortcut.1.num_batches_tracked'and k !='layer4.1.bn1.num_batches_tracked'and k !='layer4.1.bn2.num_batches_tracked':#the nn layer loop
            for i in range(1, len(w)):   #the client loop
                w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg




def p_average(w):
    w_avg = copy.deepcopy(w)
    g = 5
    h = 0.7
    for k in w_avg[0].keys():
        if k != 'bn1.num_batches_tracked' and k != 'bn2.num_batches_tracked':
        #if k !='bn1.num_batches_tracked'and k !='layer1.0.bn1.num_batches_tracked' and k !='layer1.0.bn2.num_batches_tracked' and k !='layer1.1.bn1.num_batches_tracked' and k !='layer1.1.bn2.num_batches_tracked' and k !='layer2.0.bn1.num_batches_tracked' and k !='layer2.0.bn2.num_batches_tracked' and k !='layer2.0.shortcut.1.num_batches_tracked' and k !='layer2.1.bn1.num_batches_tracked' and k !='layer2.1.bn2.num_batches_tracked'and k !='layer3.0.bn1.num_batches_tracked'and k !='layer3.0.bn2.num_batches_tracked'and k !='layer3.0.shortcut.1.num_batches_tracked'and k !='layer3.1.bn1.num_batches_tracked'and k !='layer3.1.bn2.num_batches_tracked'and k !='layer4.0.bn1.num_batches_tracked'and k !='layer4.0.bn2.num_batches_tracked'and k !='layer4.0.shortcut.1.num_batches_tracked'and k !='layer4.1.bn1.num_batches_tracked'and k !='layer4.1.bn2.num_batches_tracked':
            matrix = [[0 for i in range(10)] for i in range(10)]
            for l in range(10):
                for m in range(10):
                    matrix[l][m] = torch.mul(w_avg[l][k], w_avg[m][k]).sum().item() / (
                            torch.mul(w_avg[l][k], w_avg[l][k]).sum().item() + torch.mul(w_avg[m][k],
                                                                                         w_avg[m][
                                                                                             k]).sum().item() - torch.mul(
                        w_avg[l][k], w_avg[m][k]).sum().item())
            for l in range(10):
                for m in range(10):
                    if l == m:
                        matrix[l][m] = 0

            # print(matrix)
            sum = [0 for i in range(10)]
            for l in range(10):
                for m in range(10):
                    if l != m:
                        sum[l] += math.exp(g * matrix[l][m])

            weight = [[0 for i in range(10)] for i in range(10)]
            for p in range(10):
                for q in range(10):
                    if p != q:
                        weight[p][q] = ((math.exp(g * matrix[p][q])) / sum[p]) * h

            for p in range(10):
                for q in range(10):
                    if p == q:
                        weight[p][q] = 1 - h

            # print(weight)
            model = [0 for i in range(10)]
            for o in range(10):
                for f in range(10):
                    model[o] += torch.mul(w_avg[f][k], weight[o][f])
            for o in range(10):
                w_avg[o][k] = model[o]
        return w_avg
    #         a01 = torch.mul(w_avg[0][k], w_avg[1][k]).sum().item() / (
    #                     torch.mul(w_avg[1][k], w_avg[1][k]).sum().item() + torch.mul(w_avg[0][k],
    #                                                                                  w_avg[0][k]).sum().item() - torch.mul(
    #                 w_avg[1][k], w_avg[0][k]).sum().item())
    #         a02 = torch.mul(w_avg[0][k], w_avg[2][k]).sum().item() / (
    #                     torch.mul(w_avg[0][k], w_avg[0][k]).sum().item() + torch.mul(w_avg[2][k],
    #                                                                                  w_avg[2][k]).sum().item() - torch.mul(
    #                 w_avg[0][k], w_avg[2][k]).sum().item())
    #         a03 = torch.mul(w_avg[0][k], w_avg[3][k]).sum().item() / (
    #                     torch.mul(w_avg[0][k], w_avg[0][k]).sum().item() + torch.mul(w_avg[3][k],
    #                                                                                  w_avg[3][k]).sum().item() - torch.mul(
    #                 w_avg[0][k], w_avg[3][k]).sum().item())
    #         a04 = torch.mul(w_avg[0][k], w_avg[4][k]).sum().item() / (
    #                     torch.mul(w_avg[0][k], w_avg[0][k]).sum().item() + torch.mul(w_avg[4][k],
    #                                                                                  w_avg[4][k]).sum().item() - torch.mul(
    #                 w_avg[0][k], w_avg[4][k]).sum().item())
    #         a12 = torch.mul(w_avg[1][k],w_avg[2][k]).sum().item()/(torch.mul(w_avg[1][k], w_avg[1][k]).sum().item()+torch.mul(w_avg[2][k], w_avg[2][k]).sum().item()-torch.mul(w_avg[1][k],w_avg[2][k]).sum().item())
    #         a13 = torch.mul(w_avg[1][k], w_avg[3][k]).sum().item() / (torch.mul(w_avg[1][k], w_avg[1][k]).sum().item() + torch.mul(w_avg[3][k],w_avg[3][k]).sum().item() - torch.mul( w_avg[1][k], w_avg[3][k]).sum().item())
    #         a14 = torch.mul(w_avg[1][k], w_avg[4][k]).sum().item() / (
    #                     torch.mul(w_avg[1][k], w_avg[1][k]).sum().item() + torch.mul(w_avg[4][k],
    #                                                                                  w_avg[4][k]).sum().item() - torch.mul(
    #                 w_avg[1][k], w_avg[4][k]).sum().item())
    #         a23 = torch.mul(w_avg[2][k], w_avg[3][k]).sum().item() / (
    #                     torch.mul(w_avg[2][k], w_avg[2][k]).sum().item() + torch.mul(w_avg[3][k],
    #                                                                                  w_avg[3][k]).sum().item() - torch.mul(
    #                 w_avg[3][k], w_avg[2][k]).sum().item())
    #         a24 = torch.mul(w_avg[2][k], w_avg[4][k]).sum().item() / (
    #                     torch.mul(w_avg[2][k], w_avg[2][k]).sum().item() + torch.mul(w_avg[4][k],
    #                                                                                  w_avg[4][k]).sum().item() - torch.mul(
    #                 w_avg[4][k], w_avg[2][k]).sum().item())
    #         a34 = torch.mul(w_avg[3][k], w_avg[4][k]).sum().item() / (
    #                     torch.mul(w_avg[3][k], w_avg[3][k]).sum().item() + torch.mul(w_avg[4][k],
    #                                                                                  w_avg[4][k]).sum().item() - torch.mul(
    #                 w_avg[3][k], w_avg[4][k]).sum().item())
    #
    #     ##cos相似度
    #     # a01 = torch.mul(w_avg[0][k], w_avg[1][k]).sum().item() /(math.sqrt(torch.mul(w_avg[1][k], w_avg[1][k]).sum().item()) * math.sqrt(torch.mul(w_avg[0][k],
    #     #                                                                      w_avg[0][k]).sum().item()) )
    #     # a02 = torch.mul(w_avg[0][k], w_avg[2][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[0][k], w_avg[0][k]).sum().item() )* math.sqrt(torch.mul(w_avg[2][k],
    #     #                                                                      w_avg[2][k]).sum().item()) )
    #     # a03 = torch.mul(w_avg[0][k], w_avg[3][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[0][k], w_avg[0][k]).sum().item()) * math.sqrt(torch.mul(w_avg[3][k],
    #     #                                                                      w_avg[3][k]).sum().item()) )
    #     # a04 = torch.mul(w_avg[0][k], w_avg[4][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[0][k], w_avg[0][k]).sum().item()) * math.sqrt(torch.mul(w_avg[4][k],
    #     #                                                                      w_avg[4][k]).sum().item()) )
    #     # a12 = torch.mul(w_avg[1][k], w_avg[2][k]).sum().item() / (
    #     #             math.sqrt(torch.mul(w_avg[1][k], w_avg[1][k]).sum().item()) * math.sqrt(torch.mul(w_avg[2][k],
    #     #                                                                          w_avg[2][k]).sum().item()) )
    #     # a13 = torch.mul(w_avg[1][k], w_avg[3][k]).sum().item() / (
    #     #             math.sqrt(torch.mul(w_avg[1][k], w_avg[1][k]).sum().item()) * math.sqrt(torch.mul(w_avg[3][k],
    #     #                                                                          w_avg[3][k]).sum().item()) )
    #     # a14 = torch.mul(w_avg[1][k], w_avg[4][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[1][k], w_avg[1][k]).sum().item()) * math.sqrt(torch.mul(w_avg[4][k],
    #     #                                                                      w_avg[4][k]).sum().item()) )
    #     # a23 = torch.mul(w_avg[2][k], w_avg[3][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[2][k], w_avg[2][k]).sum().item()) * math.sqrt(torch.mul(w_avg[3][k],
    #     #                                                                      w_avg[3][k]).sum().item()) )
    #     # a24 = torch.mul(w_avg[2][k], w_avg[4][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[2][k], w_avg[2][k]).sum().item()) * math.sqrt(torch.mul(w_avg[4][k],
    #     #                                                                      w_avg[2][k]).sum().item()) )
    #     # a34 = torch.mul(w_avg[3][k], w_avg[4][k]).sum().item() / (
    #     #         math.sqrt(torch.mul(w_avg[3][k], w_avg[3][k]).sum().item()) * math.sqrt(torch.mul(w_avg[4][k],
    #     #                                                                      w_avg[4][k]).sum().item()) )
    #         w_avg[0][k] = torch.mul(w_avg[1][k], (math.exp(g * a01) / (
    #                     math.exp(g * a01) + math.exp(g * a02) + math.exp(g * a03) + math.exp(
    #                 g * a04))) * h) + torch.mul(w_avg[2][k], (math.exp(g * a02) / (
    #                     math.exp(g * a01) + math.exp(g * a02) + math.exp(g * a03) + math.exp(
    #                 g * a04))) * h) + torch.mul(w_avg[3][k], (math.exp(g * a03) / (
    #                     math.exp(g * a01) + math.exp(g * a02) + math.exp(g * a03) + math.exp(
    #                 g * a04))) * h) + torch.mul(w_avg[4][k], (math.exp(g * a04) / (
    #                     math.exp(g * a01) + math.exp(g * a02) + math.exp(g * a03) + math.exp(
    #                 g * a04))) * h) + torch.mul(w_avg[0][k], (1 - h))
    #         w_avg[1][k] = torch.mul(w_avg[0][k], (math.exp(g * a01) / (
    #                     math.exp(g * a01) + math.exp(g * a12) + math.exp(g * a13) + math.exp(
    #                 g * a14))) * h) + torch.mul(w_avg[2][k], (math.exp(g * a12) / (
    #                     math.exp(g * a01) + math.exp(g * a12) + math.exp(g * a13) + math.exp(
    #                 g * a14))) * h) + torch.mul(w_avg[3][k], (
    #                 math.exp(g * a13) / (math.exp(g * a01) + math.exp(g * a12) + math.exp(g * a13) + math.exp(
    #             g * a14))) * h) + torch.mul(w_avg[4][k], (
    #                 math.exp(g * a14) / (math.exp(g * a01) + math.exp(g * a12) + math.exp(g * a13) + math.exp(
    #             g * a14))) * h) + torch.mul(w_avg[1][k], (1 - h))
    #         w_avg[2][k] = torch.mul(w_avg[0][k], (math.exp(g * a02) / (
    #                     math.exp(g * a02) + math.exp(g * a12) + math.exp(g * a23) + math.exp(
    #                 g * a24))) * h) + torch.mul(w_avg[1][k], (
    #                 math.exp(g * a12) / (math.exp(g * a02) + math.exp(g * a12) + math.exp(g * a23) + math.exp(
    #             g * a24))) * h) + torch.mul(w_avg[3][k], (
    #                 math.exp(g * a23) / (math.exp(g * a02) + math.exp(g * a12) + math.exp(g * a23) + math.exp(
    #             g * a24))) * h) + torch.mul(w_avg[4][k], (
    #                 math.exp(g * a24) / (math.exp(g * a02) + math.exp(g * a12) + math.exp(g * a23) + math.exp(
    #             g * a24))) * h) + torch.mul(w_avg[2][k], (1 - h))
    #         w_avg[3][k] = torch.mul(w_avg[0][k], (math.exp(g * a03) / (
    #                     math.exp(g * a03) + math.exp(g * a13) + math.exp(g * a23) + math.exp(
    #                 g * a34))) * h) + torch.mul(w_avg[1][k], (
    #                 math.exp(g * a13) / (math.exp(g * a03) + math.exp(g * a13) + math.exp(g * a23) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[2][k], (
    #                 math.exp(g * a23) / (math.exp(g * a03) + math.exp(g * a13) + math.exp(g * a23) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[4][k], (
    #                 math.exp(g * a34) / (math.exp(g * a03) + math.exp(g * a13) + math.exp(g * a23) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[3][k], (1 - h))
    #         w_avg[4][k] = torch.mul(w_avg[0][k], (math.exp(g * a04) / (
    #                     math.exp(g * a04) + math.exp(g * a14) + math.exp(g * a24) + math.exp(
    #                 g * a34))) * h) + torch.mul(w_avg[1][k], (
    #                 math.exp(g * a14) / (math.exp(g * a04) + math.exp(g * a14) + math.exp(g * a24) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[2][k], (
    #                 math.exp(g * a24) / (math.exp(g * a04) + math.exp(g * a14) + math.exp(g * a24) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[3][k], (
    #                 math.exp(g * a34) / (math.exp(g * a04) + math.exp(g * a14) + math.exp(g * a24) + math.exp(
    #             g * a34))) * h) + torch.mul(w_avg[4][k], (1 - h))
    # return w_avg



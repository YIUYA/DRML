from torch import nn
import torch
from collections import OrderedDict


class multi_label_sigmoid_cross_entropy_loss(nn.Module):
    def __init__(self):
        super(multi_label_sigmoid_cross_entropy_loss, self).__init__()

    def forward(self, pred, y):
        pred = nn.Sigmoid()(pred)

        # try:
        # pos_part = (y > 0).float() * torch.log(pred)
        pos_to_log = pred[y > 0]
        pos_to_log[pos_to_log.data == 0] = 1e-20

        pos_part = torch.log(pos_to_log).sum()
        #pos_part = torch.log(pos_to_log).sum()

        # neg_part = (y < 0).float() * torch.log(1 - pred)
        #neg_to_log = 1 - pred[y < 0]
        neg_to_log = 1 - pred[y == 0]
        neg_to_log[neg_to_log.data == 0] = 1e-20
        neg_part = torch.log(neg_to_log).sum()
        #neg_part = torch.log(neg_to_log).sum()

        loss = -(pos_part + neg_part)

        loss /= pred.size(0)

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def statistics(pred, y, threshold=0.5):
    statistics_list = []
    class_au = pred.size(1)

    pred = pred > threshold

    pred = pred.long()
    batch_size = pred.size(0)
    pred[pred == 0] = 0

    for j in range(class_au):
        TP = 0
        FP = 0
        FN = 0
        TN = 0


        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False, 'pred == 1'
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False, 'pred == -1'
            else:
                assert False

        

        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})

    return statistics_list



def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)   #denominador + 1e-20
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list



def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list



def DISFA_infolist(list, action_units):
    infostr = ''
    for idx, au in enumerate(action_units):
        infostr += f'AU{au}: {list[idx] * 100.:.2f} '

    return {infostr}



def load_state_dict(model,path):

    if torch.cuda.is_available():
        checkpoints = torch.load(path,map_location=torch.device('cuda'))
    else: 
        checkpoints = torch.load(path,map_location=torch.device('cpu'))


    state_dict = checkpoints['state_dict']


    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    # load params
    model.load_state_dict(new_state_dict)

    return model


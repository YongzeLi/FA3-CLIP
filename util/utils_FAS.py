"""
Function: Utils for Unified Attack Detection
Author: Haocheng Yuan
Date: 2024/7/5

Base on:
Function: Utils for Face Anti-spoofing
Author: AJ
Date: 2021/1/15
"""
import os, cv2, time, torch, copy, sys, math, random, shutil, json, datetime
from six import iteritems
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from numpy import interp
import numpy as np
from torchvision import utils
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F

###############################################################
# Utils for Common
###############################################################
def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)

def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)

def set_env(args):
    gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    assert -1 not in gpu_ids  ### Temporarily not allowed cpu
    if len(gpu_ids) > 0:torch.cuda.set_device(gpu_ids[0])
    use_cuda = torch.cuda.is_available()
    if use_cuda and gpu_ids[0] >= 0:device = torch.device('cuda:%d' % gpu_ids[0])
    else:device = torch.device('cpu')
    return gpu_ids, use_cuda, device

def mk_jobs(args):
    ### subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    folders = []
    for folder in ['logs', 'models', 'outputs', 'scores']:
        folder = os.path.join(args.job_out, folder, args.protocol, args.subdir)
        make_if_not_exist(folder)
        folders.append(folder)
    write_args_to_file(args, os.path.join(folders[3], 'result.txt'))
    print('mkdir: logs, models, outputs, scores, result.txt')
    return folders[0], folders[1], folders[2], folders[3]

def write_args_to_file(args, filename):
    """
    :param args:
    :param filename:
    :return: write args parameter to file
    """
    with open(filename, 'a') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
def get_eta_time(num_batches, batch_idx, epoch, max_epochs, batch_time):
    nb_remain = 0
    nb_remain += num_batches - batch_idx - 1
    nb_remain += (max_epochs - epoch - 1) * num_batches
    eta_seconds = batch_time.avg * nb_remain
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    return eta

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min'%(hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec'%(min, sec)
    else:
        raise NotImplementedError

def draw_loss_crove(loss_file, L_type, net, init_lr, iter_decay_lr):
    image_file = loss_file.replace('txt', 'png')
    with open(loss_file) as f:
        all_lines = f.readlines()
    Iter = []
    Loss_c = []
    Loss_mean = []
    for iter in range(len(all_lines)-1):
        contents = all_lines[iter].strip().split('\t')
        Iter.append(iter)
        Loss_c.append(float(contents[3].split(' ')[1]))
        Loss_mean.append(float(contents[4].split(' ')[1]))
    figsize = 20, 10
    figure, ax = plt.subplots(figsize=figsize)
    lw = 2
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlim([0.0, iter])
    plt.ylim([0.0, Loss_c[0]])
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Iter', font2)
    plt.ylabel('Loss', font2)
    plt.title('Iter vs Loss(L{}): net={} init_lr={} iter_decay_lr={}'.format(L_type, net, init_lr, iter_decay_lr), font2)
    plt.plot(Iter, Loss_c, color='red', lw=lw, label='loss_real_time')
    plt.plot(Iter, Loss_mean, color='blue', lw=lw, label='Acc')
    plt.legend(loc="upper right", prop=font1)
    plt.savefig(image_file)
    # plt.show()

def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)

def save_image(tensor, outdir, epoch_iter, prefix='XY', nrow=4):
    filename = os.path.join(outdir, '{}_iter_{}.jpg'.format(str(epoch_iter), prefix))
    utils.save_image(tensor, filename, nrow=int(nrow), normalize=True)

def cutmix_images_slide(tensor_x, tensor_y, stride, grid_size, patch_size, cutmixNum=0, cutmixIndexs=None):
    """
    :param grid_size:
    :param patch_size:
    :param tensor: [-1, 3, h, w]
    :return: tensor_x, tensor_y
    """
    if cutmixNum == 0:
        return tensor_x, tensor_y, cutmixIndexs
    index = 0
    patch_x = torch.zeros([tensor_x.shape[0], tensor_x.shape[1], patch_size, patch_size])
    patch_y = torch.zeros([tensor_y.shape[0], tensor_y.shape[1], patch_size, patch_size])

    mixupIndexs = random.sample(range(grid_size * grid_size), cutmixNum) if cutmixIndexs is None else cutmixIndexs
    for i in range(grid_size):
        for j in range(grid_size):
            patch_x[:,:,:,:] = tensor_x[:, :, stride*i:stride*i+patch_size, stride*j:stride*j+patch_size]
            patch_y[:,:,:,:] = tensor_y[:, :, stride*i:stride*i+patch_size, stride*j:stride*j+patch_size]
            if index in mixupIndexs:
                tensor_x[:, :, stride * i:stride * i + patch_size, stride * j:stride * j + patch_size] = patch_y
                tensor_y[:, :, stride * i:stride * i + patch_size, stride * j:stride * j + patch_size] = patch_x
            index += 1
    return tensor_x, tensor_y, mixupIndexs


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  ### if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            ### if the buffer is not full; keep inserting current images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                ### by 50% chance, the buffer will return a previously stored image,
                ### and insert the current image into the buffer
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def adjust_lr(args, optimizer, epoch, lr_decay_epochs, lr_decay=0.1):
    lr = args.lr
    if args.warm_cosine[1]:
        eta_min = lr * (lr_decay ** 4)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.max_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
        # for s in lr_decay_epochs:
        #     if epoch >= s:
        #         lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warmup_learning_rate(args, lr_c, batch_id, epoch, total_batches, optimizer):
    # warm-up for large-batch training
    warm_epochs = 1
    if args.warm_cosine[1]:
        eta_min = args.lr * (args.lr_decay_rate ** 3)
        warmup_to = eta_min + \
                    (args.lr - eta_min) * \
                    (1 + math.cos(math.pi * warm_epochs / args.max_epochs)) / 2
    else:
        warmup_to = args.lr

    ### update lr
    warmup_from = 0.01
    if args.batch_size > 256:warm = True
    else:warm = False
    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('update lr by warm-up', 'epoch:{}, batch_id:{}, lr:{}'.format(epoch, batch_id, lr))
        return lr
    else:
        return lr_c


###############################################################
# Test and Metric:
###############################################################
def split_list(file_list_1, file_list_2, wanted_parts=1):
    length = len(file_list_1)
    block_1 = [file_list_1[i * length // wanted_parts: (i + 1) * length // wanted_parts] for i in range(wanted_parts)]
    block_2 = [file_list_2[i * length // wanted_parts: (i + 1) * length // wanted_parts] for i in range(wanted_parts)]
    return block_1, block_2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TP = np.sum((labels == 1) & (predict == True))
    FN = np.sum((labels == 1) & (predict == False))
    TN = np.sum((labels == 0) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    return TN, FN, FP, TP

def get_thr(probs, labels, grid_density=10000):
    def get_threshold(grid_density):
        thresholds = []
        for i in range(grid_density + 1):
            thresholds.append(0.0 + i * 1.0 / float(grid_density))
        thresholds.append(1.1)
        return thresholds
    thresholds = get_threshold(grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return thr

def compute_eer(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold

def get_Metrics_at_thr(probs, labels, thr):
    AUC = roc_auc_score(labels, probs)
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif (FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    ACC = (TP + TN) / labels.shape[0]
    test_fpr, test_tpr, test_thre = roc_curve(copy.deepcopy(labels), copy.deepcopy(probs), pos_label=1)
    Recall2 = interp(0.01, test_fpr, test_tpr)
    APCER = FAR
    BPCER = FRR
    ACER = (APCER + BPCER) / 2.0
    EER, EER_threshold = compute_eer(probs, labels)

    return HTER, AUC, Recall2, ACC, ACER, APCER, BPCER, EER

def cross_entropy(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob
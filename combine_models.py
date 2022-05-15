
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import re

import os

#import train_and_test as tnt
import time
from util.preprocess import mean, std


class PPNet_ensemble(nn.Module):

    def __init__(self, ppnets):
        super(PPNet_ensemble, self).__init__()
        self.ppnets = ppnets  # a list of ppnets

    def forward(self, x):
        logits, min_distances_0 = self.ppnets[0](x)
        min_distances = [min_distances_0]
        for i in range(1, len(self.ppnets)):
            logits_i, min_distances_i = self.ppnets[i](x)
            logits.add_(logits_i)
            min_distances.append(min_distances_i)
        return logits, min_distances


##### MODEL AND DATA LOADING
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# load the models
# provide paths to saved models you want to combine:
# e.g. load_model_paths = ['./saved_models/densenet121/003/30_18push0.8043.pth',
#                          './saved_models/resnet34/002/10_19push0.7920.pth',
#                          './saved_models/vgg19/003/10_18push0.7822.pth']
# MUST NOT BE EMPTY

load_model_paths = []
ppnets = []
epoch_number_strs = []
start_epoch_numbers = []

for load_model_path in load_model_paths:
    load_model_name = load_model_path.split('/')[-1]
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    epoch_number_strs.append(epoch_number_str)

    start_epoch_number = int(epoch_number_str)
    start_epoch_numbers.append(start_epoch_number)

    print('load model from ' + load_model_path)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnets.append(ppnet)

ppnet_ensemble = PPNet_ensemble(ppnets)
ppnet_ensemble = ppnet_ensemble.cuda()
ppnet_ensemble_multi = torch.nn.DataParallel(ppnet_ensemble)

img_size = ppnets[0].img_size

# ppnet_multi = torch.nn.DataParallel(ppnet)
# img_size = ppnet_multi.module.img_size
# prototype_shape = ppnet.prototype_shape
# max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

# load the (test) data
from settings_CUB import test_dir

test_batch_size = 80

normalize = transforms.Normalize(mean=mean,
                                 std=std)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
print('test set size: {0}'.format(len(test_loader.dataset)))


for ppnet in ppnet_ensemble_multi.module.ppnets:
    print(ppnet)

class_specific = True


# only supports last layer adjustment
def _train_or_test_ppnet_ensemble(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                                  coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, _ = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            l1 = torch.tensor(0.0).cuda()

            if class_specific:
                if use_l1_mask:
                    for ppnet in model.module.ppnets:
                        l1_mask = 1 - torch.t(ppnet.prototype_class_identity).cuda()
                        l1_ = (ppnet.last_layer.weight * l1_mask).norm(p=1).cuda()
                        l1.add_(l1_)
                else:
                    for ppnet in model.module.ppnets:
                        l1_ = ppnet.last_layer.weight.norm(p=1)
                        l1.add_(l1_)

            else:
                for ppnet in model.module.ppnets:
                    l1_ = ppnet.last_layer.weight.norm(p=1)
                    l1.add_(l1_)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['i_crs_ent'] * cross_entropy
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['i_crs_ent'] * cross_entropy
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    last_layer_p1_norm = 0
    for ppnet in model.module.ppnets:
        last_layer_p1_norm += ppnet.last_layer.weight.norm(p=1).item()
    log('\tl1: \t\t{0}'.format(last_layer_p1_norm))
    # p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    # with torch.no_grad():
    #    p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train_ensemble(model, dataloader, optimizer, class_specific=True, coefs=None, log=print):
    assert (optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=optimizer,
                                         class_specific=class_specific, coefs=coefs, log=log)


def test_ensemble(model, dataloader, class_specific=True, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=None,
                                         class_specific=class_specific, log=log)


def ensemble_last_only(model, log=print):
    for ppnet in model.module.ppnets:
        for p in ppnet.features.parameters():
            p.requires_grad = False
        for p in ppnet.add_on_layers.parameters():
            p.requires_grad = False
        ppnet.prototype_vectors.requires_grad = False
        for p in ppnet.last_layer.parameters():
            p.requires_grad = True
    log('\tensemble last layer')


#check test accuracy
accu = test_ensemble(model=ppnet_ensemble_multi, dataloader=test_loader,
                     class_specific=class_specific, log=print)



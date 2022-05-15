import time
import torch
import torch.nn.functional as F

from util.helpers import list_of_distances


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
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
    total_cluster_cost = 0
    total_orth_cost = 0
    total_subspace_sep_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0


    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()
        #with autograd.detect_anomaly():
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)
            del input
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])#

                subspace_max_dist = (model.module.prototype_shape[0]* model.module.prototype_shape[2]* model.module.prototype_shape[3]) #2000
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                #optimize orthogonality of prototype_vector
                cur_basis_matrix = torch.squeeze(model.module.prototype_vectors) #[2000,128]
                subspace_basis_matrix = cur_basis_matrix.reshape(model.module.num_classes,model.module.num_prototypes_per_class,model.module.prototype_shape[1])#[200,10,128]
                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2) #[200,10,128]->[200,128,10]
                orth_operator = torch.matmul(subspace_basis_matrix,subspace_basis_matrix_T)  # [200,10,128] [200,128,10] -> [200,10,10]
                I_operator = torch.eye(subspace_basis_matrix.size(1),subspace_basis_matrix.size(1)).cuda() #[10,10]
                difference_value = orth_operator - I_operator #[200,10,10]-[10,10]->[200,10,10]
                orth_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0)) #[200]->[1]

                del cur_basis_matrix
                del orth_operator
                del I_operator
                del difference_value

                #subspace sep
                projection_operator = torch.matmul(subspace_basis_matrix_T,subspace_basis_matrix)#[200,128,10] [200,10,128] -> [200,128,128]
                del subspace_basis_matrix
                del subspace_basis_matrix_T

                projection_operator_1 = torch.unsqueeze(projection_operator,dim=1)#[200,1,128,128]
                projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)#[1,200,128,128]
                pairwise_distance =  torch.norm(projection_operator_1-projection_operator_2+1e-10,p='fro',dim=[2,3]) #[200,200,128,128]->[200,200]
                subspace_sep = 0.5 * torch.norm(pairwise_distance,p=1,dim=[0,1],dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()
                del projection_operator_1
                del projection_operator_2
                del pairwise_distance


                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    # weight 200,2000   prototype_class_identity [2000,200]
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)
            else:
                cluster_cost = torch.Tensor([0])
                separation_cost = torch.Tensor([0])
                l1 = torch.Tensor([0])
                orth_cost = torch.Tensor([0])
                subspace_sep = torch.Tensor([0])
                separation_cost = torch.Tensor([0])
                avg_separation_cost = torch.Tensor([0])


            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_orth_cost += orth_cost.item()
            total_subspace_sep_cost += subspace_sep.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()


        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + coefs['orth'] * orth_cost
                          + coefs['sub_sep'] * subspace_sep)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1 * orth_cost - 1e-7 * subspace_sep
            #print("{}/{} loss:{} cre:{} clst:{} sep:{} l1:{} orth:{} sub_sep:{}".format(i,len(dataloader),loss,cross_entropy,cluster_cost,separation_cost,l1,orth_cost,subspace_sep))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #nomalize basis vectors
            model.module.prototype_vectors.data = F.normalize(model.module.prototype_vectors, p=2, dim=1).data



        #del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\torth: \t{0}'.format(total_orth_cost / n_batches))
    log('\tsubspace_sep: \t{0}'.format(total_subspace_sep_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'cluster_loss': total_cluster_cost / n_batches,
                    'orth_loss': total_orth_cost / n_batches,
                    'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'avg_separation': total_avg_separation_cost / n_batches,
                    'l1':model.module.last_layer.weight.norm(p=1).item(),
                    'p_avg_pair_dist':p_avg_pair_dist,
                    'accu' : n_correct/n_examples
                    }
    return n_correct / n_examples,results_loss


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')

    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = False
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')

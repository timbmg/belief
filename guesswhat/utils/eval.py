import torch
from tqdm import tqdm


def eval_epoch(model, data_loader, forward_kwargs_mapping, target_kwarg,
               loss_fn, optimizer=None, clip_norm_args=None,
               manipulate_targets=None):

    epoch_loss, epoch_acc = 0, 0

    if optimizer is not None:
        model.train()
        torch.enable_grad()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
    else:
        model.eval()
        torch.no_grad()

    desc = 'train' if optimizer is not None else 'valid'
    with tqdm(total=len(data_loader), desc=desc, unit='batches') as pbar:
        for iteration, batch in enumerate(data_loader):

            model_kwargs = dict()
            for model_key, batch_key in forward_kwargs_mapping.items():
                model_kwargs[model_key] = batch[batch_key]

            logits = model(**model_kwargs)

            if manipulate_targets is not None:
                batch[target_kwarg] = manipulate_targets(batch[target_kwarg])

            loss = loss_fn(logits, batch[target_kwarg].view(-1))

            epoch_loss += loss.item()
            epoch_acc += accuarcy(logits, batch[target_kwarg])

            if optimizer is not None:
                for opti in optimizer:
                    opti.zero_grad()
                loss.backward()
                for opti in optimizer:
                    if clip_norm_args is not None:
                        for pg in range(len(opti.param_groups)):
                            torch.nn.utils.clip_grad_norm_(
                                opti.param_groups[pg]['params'], *clip_norm_args)
                    opti.step()

            pbar.update(1)

        return epoch_loss/len(data_loader), epoch_acc/len(data_loader)


def accuarcy(logits, targets):

    return torch.eq(logits.topk(1)[1].view(-1),
                    targets.view(-1)).sum().item() / targets.size(0)


def multi_target_accuracy(logits, targets):

    y = logits.topk(1)[1].view(-1).unsqueeze(1)
    one_hot_y = logits.new_empty(logits.size()).fill_(0)
    one_hot_y.scatter_(1, y, 1)
    one_hot_y = one_hot_y.float()

    return (torch.sum(one_hot_y*targets) / targets.size(0)).item()

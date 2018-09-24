import torch


def eval_epoch(model, data_loader, forward_kwargs_mapping, target_kwarg,
               loss_fn, optimizer=None, logger=None):

    epoch_loss, epoch_acc = 0, 0

    if optimizer is not None:
        model.train()
        torch.enable_grad()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
    else:
        model.eval()
        torch.no_grad()

    for iteration, batch in enumerate(data_loader):

        model_kwargs = dict()
        for model_key, batch_key in forward_kwargs_mapping.items():
            model_kwargs[model_key] = batch[batch_key]

        logits = model(**model_kwargs)

        loss = loss_fn(logits, batch[target_kwarg].view(-1))

        epoch_loss += loss.item()
        epoch_acc += accuarcy(logits, batch[target_kwarg])

        if optimizer is not None:
            for opti in optimizer:
                opti.zero_grad()
            loss.backward()
            for opti in optimizer:
                opti.step()

    return epoch_loss/len(data_loader), epoch_acc/len(data_loader)


def accuarcy(logits, targets):

    return torch.eq(logits.topk(1)[1].view(-1),
                    targets.view(-1)).sum().item() / targets.size(0)

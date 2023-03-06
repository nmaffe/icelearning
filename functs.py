import torch

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(self.sum / self.count)

    def update_simplesum(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(self.sum / self.count)


def train_fn(train_loader, model, criterion, optimizer, scheduler, device):
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()

    for step, (images, labels, paths, xfeatures) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        xfeatures = xfeatures.to(device)

        # with torch.set_grad_enabled(True):
        y_preds = model(images, xfeatures)
        loss = criterion(y_preds, labels)
        preds = (y_preds == y_preds.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

        # statistics
        losses.update(loss.item(), images.size(0))
        how_many_correct = torch.sum(torch.all(torch.eq(preds, labels), dim=1))
        accuracies.update_simplesum(how_many_correct.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    print(f'Train Loss: {losses.avg:.4f} Acc: {accuracies.avg:.4f}')

    return losses.history, accuracies.history


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()

    for step, (images, labels, paths, xfeatures) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        xfeatures = xfeatures.to(device)

        # compute loss
        with torch.no_grad():
            y_preds = model(images, xfeatures)
            loss = criterion(y_preds, labels)  
            preds = (y_preds == y_preds.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

        # statistics
        losses.update(loss.item(), images.size(0))
        how_many_correct = torch.sum(torch.all(torch.eq(preds, labels), dim=1))
        accuracies.update_simplesum(how_many_correct.item(), images.size(0))

    print(f'Val Loss: {losses.avg:.4f} Acc: {accuracies.avg:.4f}')

    return losses.history, accuracies.history


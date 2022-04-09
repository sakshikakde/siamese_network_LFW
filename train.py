from common_utils import *

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    train_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        train_loss += loss.item()
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_interval = 50
        if (batch_idx + 1) % log_interval == log_interval-1:
            avg_loss = train_loss / log_interval
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                (batch_idx + 1) * len(inputs), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), avg_loss))
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(train_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg
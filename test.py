from common_utils import *

def test(model, device, test_loader, criterion):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  

            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

    # show info
    print('Test set set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(test_loader.dataset), losses.avg, accuracies.avg * 100))
        
    return losses.avg, accuracies.avg
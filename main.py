from common_utils import *
from data_utils import *
from dataset.lfw import LFW_dataset
from models.siamese_model import SiameseNet as Net
from train import *
from validate import *
from test import *

def log_data_info(train_dataloader, val_dataloader, test_dataloader):
    print("input dimensions are:", train_dataloader.dataset[0][0].shape)
    print("Train set : ", len(train_dataloader.dataset))
    print("Validation set : ", len(val_dataloader.dataset))
    print("Test set : ", len(test_dataloader.dataset))

    print("---------------------------------------")
    print("Data distribution in training set:")
    train_distribution = torch.zeros(2, dtype=torch.float)
    for _, label in train_dataloader.dataset:
        train_distribution += label
    print("Different person: ", train_distribution.numpy()[0])
    print("Same person: ", train_distribution.numpy()[1])

    print("---------------------------------------")
    print("Data distribution in val set:")
    val_distribution = torch.zeros(2, dtype=torch.float)
    for _, label in val_dataloader.dataset:
        val_distribution += label
    print("Different person: ", val_distribution.numpy()[0])
    print("Same person: ", val_distribution.numpy()[1])

    print("---------------------------------------")
    print("Data distribution in test set:")
    test_distribution = torch.zeros(2, dtype=torch.float)
    for _, label in test_dataloader.dataset:
        test_distribution += label
    print("Different person: ", test_distribution.numpy()[0])
    print("Same person: ", test_distribution.numpy()[1])
    print("---------------------------------------")
    return train_distribution, val_distribution, test_distribution


def main():
    batch_size = 32
    num_epoch = 10 
    save_interval = 2
    use_cuda = torch.cuda.is_available()
    device = "cpu" #torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {} 

    print("Using device ", device)
    save_path = "./snapshots/"
    model_name = "siamese_conv4_CE"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Folder to save models created.")

    if not os.path.exists(os.path.join(save_path, model_name)):
        os.mkdir(os.path.join(save_path, model_name))

    data_file_name = "./data/images_list.txt"
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(data_file_name, batch_size, transform)
    train_distribution, _, _ = log_data_info(train_dataloader, val_dataloader, test_dataloader)

    model = Net().to(device)

    '''Start Training'''
    best_val_accuracy = 0
    best_state = None
    best_epoch = None
    best_val_loss = None

    class_weights = torch.Tensor([len(train_dataloader.dataset) / train_distribution.numpy()[0],
                              len(train_dataloader.dataset) / train_distribution.numpy()[1]])

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='mean')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(0, num_epoch):
        print("------------------------- epoch : ", epoch, " -------------------------")
        train_loss, train_acc = train_epoch(model, device, train_dataloader, optimizer, criterion)
        val_loss, val_acc = val_epoch(model, device, val_dataloader, criterion)
        test_loss, test_acc = test_epoch(model, device, test_dataloader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        if(val_acc >= best_val_accuracy):
            best_state = state
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_accuracy = val_acc

        if (epoch) % save_interval == save_interval-1:
            timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            torch.save(best_state, os.path.join(os.path.join(save_path, model_name), f'{model_name}-Epoch-{epoch}-Loss-{val_loss}_{timestamp}.pth'))
            print("Model saved with val accuracy ", val_acc)

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    torch.save(best_state, os.path.join(save_path, model_name, f'best_model-{model_name}-Epoch-{epoch}-Loss-{val_loss}_{timestamp}.pth'))
    print("best model saved with val accuracy ", val_acc)

    # plots
    plot_loss(train_losses, "Training loss")
    plot_loss(val_losses, "validation loss")
    plot_loss(test_losses, "Testing loss")
    plot_accuracy(train_accuracies, "Training accuracy")
    plot_accuracy(val_accuracies, "Validation accuracy")
    plot_accuracy(test_accuracies, "Testing accuracy")
    plt.show()
    

if __name__ == "__main__":
	main()
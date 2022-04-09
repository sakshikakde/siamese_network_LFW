from common_utils import *
from data_utils import *
from dataset.lfw import LFW_dataset

def log_data_info(train_dataloader, val_dataloader, test_dataloader):
    print("input dimensions are:", train_dataloader.dataset[0][0].shape)
    print("Train set : ", len(train_dataloader.dataset))
    print("Validation set : ", len(val_dataloader.dataset))
    print("Test set : ", len(test_dataloader.dataset))

    print("---------------------------------------")
    print("Data distribution in training set:")
    distribution = torch.zeros(2, dtype=torch.float)
    for _, label in train_dataloader.dataset:
        distribution += label
    print("Different person: ", distribution.numpy()[0])
    print("Same person: ", distribution.numpy()[1])

    print("---------------------------------------")
    print("Data distribution in val set:")
    distribution = torch.zeros(2, dtype=torch.float)
    for _, label in val_dataloader.dataset:
        distribution += label
    print("Different person: ", distribution.numpy()[0])
    print("Same person: ", distribution.numpy()[1])

    print("---------------------------------------")
    print("Data distribution in test set:")
    distribution = torch.zeros(2, dtype=torch.float)
    for _, label in test_dataloader.dataset:
        distribution += label
    print("Different person: ", distribution.numpy()[0])
    print("Same person: ", distribution.numpy()[1])


def main():
    batch_size = 32
    num_epoch = 10 
    save_interval = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_file_name = "images_list.txt"
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(data_file_name, transform)
    log_data_info(train_dataloader, val_dataloader, test_dataloader)

    

if __name__ == "__main__":
	main()
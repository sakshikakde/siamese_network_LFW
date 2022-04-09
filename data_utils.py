from common_utils import *
from dataset.lfw import LFW_dataset

def get_image_ids(file_name):
    images_dict = {}
    count = 0
    for line in open(file_name,"r"):
        line = line.strip()
        person = line.split("/")[-2]
        if person not in images_dict:
            images_dict[person] = [line]
        else:
            images_dict[person].append(line)
        count += 1

    print("Number of unique persons = ", str(len(images_dict)))
    print("NUmber of total images = ", str(count))
    unique_ids = list(images_dict.keys())
    val_ids = unique_ids[-800:-400]
    test_ids = unique_ids[-400:]
    train_ids = unique_ids[:-800]
    return images_dict, train_ids, val_ids, test_ids

def get_data_loaders(data_file_name, batch_size, transform = None):
    images_dict, train_ids, val_ids, test_ids = get_image_ids(data_file_name) 

    train_dataset = LFW_dataset(images_dict, ids = train_ids, split="train", transform = transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    val_dataset = LFW_dataset(images_dict, ids = val_ids, split="val", transform = transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = LFW_dataset(images_dict, ids = test_ids, split="test", transform = transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size) 

    return train_dataloader, val_dataloader, test_dataloader
from common_utils import *
from models.siamese_model import SiameseNet as Net
from predict import *
from data_utils import *

def load_model(model_path, model, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

def main():
    model_path = "./snapshots/siamese_conv4_CE/best_model-siamese_conv4_CE-Epoch-49-Loss-16.17410554885864_Apr-08-2022_2237.pth"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    load_model(model_path, model, device)

    data_file_name = "./data/images_list.txt"
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    batch_size = 32

    _, _, test_dataloader = get_data_loaders(data_file_name, batch_size, transform)


    num_images = 5
    fig, axs = plt.subplots(num_images, 2, figsize=(10 ,5 * num_images))
    for i in range(num_images):
        idx = random.randint(0, len(test_dataloader))
        data = test_dataloader.dataset[idx]
        images = data[0]
        actual_label = data[1]
        print_label = actual_label
        pred_label = predict(model, images.unsqueeze(0).to(device))
        if(pred_label == 0):
            print_label = np.array([1, 0])
        else:
            print_label = np.array([0, 1])


        image1 = images[0:3, :, :].swapaxes(0, 2).swapaxes(0, 1)
        image2 = images[3:6, :, :].swapaxes(0, 2).swapaxes(0, 1)

        axs[i, 0].imshow(image1)
        axs[i, 0].text(2, 15, r'actual label : ' + str(actual_label.detach().numpy()), fontsize=20, color='green')
        axs[i, 1].imshow(image2)
        axs[i, 1].text(2, 15, r'predicted label : ' + str(print_label), fontsize=20, color='red')
    
    plt.show()

if __name__ == "__main__":
	main()
    



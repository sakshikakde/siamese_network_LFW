import os
import glob

data_folder = "./data/"
image_file_name = os.path.join(data_folder, 'images_list.txt')
txt_file = open(image_file_name, "w")
files = glob.glob(os.path.join(data_folder + 'lfw/**/*.jpg'), recursive=True)
print("Writing to ", image_file_name, ":", len(files))
for file in files:
    txt_file.writelines(file + "\n")
txt_file.close()
# siamese network for LFW 
## Data generation

1) Download the data from [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and extract it to 'data' folder
2) Run the following command from the root directory:
``` python3 generate_data.py ```     

A 'images_list.txt' file will be created in 'data' folder.

## Training
1) Run the following command:    
``` python3 main.py```
2) The training will begin and model files will be saved in 'snapshot' folder. A best model file will be saved based on the best validation accuracy achieved.

### Model
A pretrained model can be downloaded from [here](https://drive.google.com/file/d/10Dawy1RakjSFz786Xny6TTSqWk8lD329/view?usp=sharing).


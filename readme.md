# Audio Steganalysis with CNN
@ author: **Wang Yuntao (Charles_wyt)** <br>
This project is a tensorflow implementation of recent work [CNN-based Steganalysis of MP3 Steganography in the Entropy
Code Domain](http://www.media-security.net/?p=809).
## Necessary Package
tensorflow-gpu==1.3 or later, numpy, pandas, matplotlib, scikit-image, scikit-learn, librosa(depend on FFmpeg)

## The architecture of the network

![The structure of the proposed network](https://i.imgur.com/h0o5lfB.jpg)

## Steganographic Algorithm
**HCM** (Huffman Code Mapping) and **EECS** (Equal Length Entropy Codes Substitution, an adaptive MP3 steganographic algorithm with STC and distortion function based on psychological acoustics model (PAM))

Note: All built-in MP3 algorithms embeds secret messages in the process of MP3 encoding, which will change QMDCT coefficients of MP3. So, this network can be applied to detect all data hiding methods which have impact on the QMDCT coefficients.

## Dataset
The dataset can be downloaded from [**Audio Steganalysis Dataset, Institute of Information Engineering (ASDIIE)**](https://pan.baidu.com/s/1ZRkfQTBXg4qMrASR_-ZBSQ) <br>
The *extraction password* is "**1fzi**".

The information of Dataset is listed as:


## File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
01      |   audio_preprocess.py     |   include some pre-process methods for **audio**
02      |   text_preprocess.py      |   include some pre-process methods for **text**
03      |   image_preprocess.py     |   include some pre-process methods for **image** 
04      |   file_preprocess.py      |   get the name, size and type of the **file**
05      |   samples_make.py         |   samples make script
06      |   pre_process.py          |   some pre-processing method such as **truncation**, **down_sampling**
07      |   classifier.py           |   machine learning classifiers such as **SVM**, **KNN**, and **model selection**, **ROC** plot, etc.
08      |   config.py               |   **command parser** and some package management
09      |   filters.py              |   some **filters** used for pre-processing such as kv kernel or other **rich model**
10      |   **main.py**             |   the main program
11      |   manager.py              |   **GPU** management (free GPU selection **automatically**)
12      |   layer.py                |   basic unit in CNN such as **conv layer**, **pooling layer**, **BN layer** and so on
13      |   network.py              |   various networks including **VGG19**, **LeNet** and **ourselves' network**
14      |   utils.py                |   some useful tools such as **minibatch**, **get_model_info**, 
15      |   run.py                  |   the **train** and **test** of the network **get_weights**, **get_biases** and so on
16      |   TODO                    |   to do list (used by myself)

## Run
* install **python3.x** or **Anaconda** and add the path into the environment variable
* GPU run environment configure if train the network (optional)
* pip install **tensorflow==1.3 or later, numpy, pandas, scikit-learn, scikit-image, librosa** (depend on FFmpeg, optional)
* run the code as the example as follows
* use tensorboard to visualize the train process such as the accuracy and loss curve of train and valid. The command is "tensorboard --logdir=/path/to/log-directory"

## Command Parser
Command: (sudo) python3(.5) main.py --argument 1 --argument 2 ... --argument N <br>

## How to use
* Example(**train**): **sudo python3.5 main.py** --**mode** train --**network** network1 --**gpu_selection** auto --**width** 380 --**is_diff** True --**order** 2 --**direction** 0 --**cover_valid_path** xxx --**cover_valid_path** xxx --**stego_train_path** xxx --**stego_valid_path** xxx --**models_path** xxx --**logs_path** xxx

* Example(**test**): **sudo python3.5 main.py** --**mode** test --**network** network1 --**width** 380 --**batch_size_test** 64 --**is_diff** True --**order** 2 --**direction** 0 --**model_files_path** xxx --**cover_test_path** xxx --**stego_test_path** xxx

* Example(**steganalysis**): **sudo python3.5 main.py** --**mode** steganalysis --**submode** one --**network** network1 --**width** 380 --**is_diff** True --**order** 2 --**direction** 0 --**file_path** xxx --**model_file_path** xxx

* Example(**steganalysis**): **sudo python3.5 main.py** --**mode** steganalysis --**submode** batch --**network** network1 --**width** 380 --**is_diff** True --**order** 2 --**direction** 0 --**files_path** xxx --**label_file_path** xxx --**model_file_path** xxx

### Something to be detailed
1. Copy the command line and modify the file path according to your configuration, more descriptions of each variant can be seen in the **config.py**.
2. When you load the model, if you choose the parameter **model_file_path**, write "audio_steganalysis-45000" as the file path, if you choose the parameter **model_files_path**, write the models file path which consists of trained models.
3. The parameter label_file_path is optional, if the parameter is not selected, all labels are None.
4. Up to now, the code can be used for audio and image, you can choose the type of carrier via the parameter **carrier**.
5. The default mode of GPU selection is "auto", the code also can run with CPU or GPU directly.
6. If you use the trained model, modify the path in checkpoint accordingly.

Note, the type of pre-processing method must be the **same** at the stage of train, valid, test and steganalysis.

## The description of each network
*  **network for audio steganalysis**

        network1   : The proposed network (最终选定

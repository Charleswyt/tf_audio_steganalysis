# Audio Steganalysis with CNN
@ author: **Wang Yuntao (Charles_wyt)** <br>
This project is a tensorflow implementation of recent work [CNN-based Steganalysis of MP3 Steganography in the Entropy
Code Domain](http://www.media-security.net/?p=809). Hope we can have a friendly communication.
Email: wangyuntao2@iie.ac.cn

## Necessary Package
tensorflow-gpu==1.3 or later, numpy, pandas, matplotlib, scikit-image, scikit-learn, librosa(depend on FFmpeg)

## The architecture of the network

![The structure of the proposed network](https://i.imgur.com/h0o5lfB.jpg)

## Steganographic Algorithm
**HCM** (Huffman Code Mapping) and **EECS** (Equal Length Entropy Codes Substitution, an adaptive MP3 steganographic algorithm with STC and distortion function based on psychological acoustics model (PAM))

Note: All built-in MP3 algorithms embeds secret messages in the process of MP3 encoding, which will change QMDCT coefficients of MP3. So, this network can be applied to detect all data hiding methods which have impact on the QMDCT coefficients.

## Dataset
The dataset can be downloaded from [**Audio Steganalysis Dataset, IIE (ASDIIE)**](https://pan.baidu.com/s/1ZRkfQTBXg4qMrASR_-ZBSQ) <br>
The *extraction password* is "**1fzi**".

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

        network1   : The proposed network (最终选定的网络)
        network1_1 : Remove all BN layers (去掉所有BN层)
        network1_2 : Average pooling layer is used for subsampling (将所有的降采样方式改为平均池化方式)
        network1_3 : Convolutional layer with stride 2 is used for subsampling (将所有的降采样方式改为卷积池化方式)
        network1_4 : Replace the convolutional kernel with 5x5 kernel (将卷积核尺寸由3 x 3改为5 x 5)
        network1_5 : ReLu is used as the activation function (将激活函数由Tanh改为ReLu)
        network1_6 : Leaky-ReLu is used as the activation function (将激活函数由tanh改为Leaky-ReLu)
        network1_7 : Deepen the network to block convolution layers (加深网络)
        network1_8 : Design a network to steganalyze audios of arbitrary size (解决可变尺寸输入数据的训练问题)
        network1__1: Remove the BN layer in the first group (去除第一个卷积块中的BN层)
        network1__2: Remove the BN layers in the first two groups (去除前两个卷积块中的BN层)
        network1__4: Remove the BN layers in the first four groups (去除前四个卷积块中的BN层)

        Note: HPF and ABS is applied at the pre-processing
    
* **network for image steganalysis**
    
        stegshi   : Xu-Net

    
## The method of pre-processing
    
There are positive and negative values in QMDCT coefficients matrix. The values in interval **[-15, 15]** is modified.
The ratio of values in **[-15, 15]** is more than **99%**, as the figure shown. <br>

* Abs <br>
* Truncation <br>
* Down-sampling <br>
![The distribution of QMDCT coefficients](https://i.imgur.com/vDJ2gWm.jpg)
The proportion of zero value is far greater than others, so we remove the zero value in the right figure for better visual effects.
## Reference
**[1]** Haiying Gao. 2007. **The MP3 steganography algorithm based on Huffman coding**. Acta Scientiarum Naturalium Universitatis Sunyatseni 4 (2007), 009. <br>
**[2]** Diqun Yan, Rangding Wang, and Liguang ZHANG. 2011. **A high capacity MP3 steganography based on Huffman coding**. Journal of Sichuan University (Natural Science Edition) 6 (2011), 013. <br>
**[3]** Kun Yang, Xiaowei Yi, Xianfeng Zhao, and Linna Zhou. 2017. **Adaptive MP3 Steganography Using Equal Length Entropy Codes Substitution**. In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 202–216. <br>
**[4]** Yanzhen Ren, Qiaochu Xiong, and Lina Wang. 2017. **A Steganalysis Scheme for AAC Audio Based on MDCT Difference Between Intra and Inter Frame**. In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 217–231. <br>
**[5]** Chao Jin, Rangding Wang, and Diqun Yan. 2017. **Steganalysis of MP3Stego with low embedding-rate using Markov feature**. Multimedia Tools and Applications 76, 5 (2017), 6143–6158. <br>
**[6]** **tensorflow** API: https://www.tensorflow.org/ <br>
**[7]** **tensorlayer** API: http://tensorlayer.readthedocs.io/en/latest/ <br>
**[8]** **tensorboard** usage: http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/graph_viz.html <br>
**[9]** **FFmpeg**: http://www.ffmpeg.org/download.html <br>
**[10]** **Python**: https://www.python.org/ <br>
**[11]** **Anaconda**: https://repo.continuum.io/archive/ <br>
**[12]** **librosa API**: http://librosa.github.io/librosa/core.html <br>
**[13]** **librosa introduction (Chinese)**: https://www.cnblogs.com/xingshansi/p/6816308.html <br>

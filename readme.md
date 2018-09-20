# Audio and Image Steganalysis with CNN
@ Author: **Wang Yuntao (Charles_wyt)** <br>
@ Email: wangyuntao2@iie.ac.cn <br>
Hope we can have a friendly communication.

This project is a tensorflow implementation of recent work, you can also design your own network via this platform.
+ [CNN-based Steganalysis of MP3 Steganography in the Entropy
Code Domain](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/CNN-based%20Steganalysis%20of%20MP3%20Steganography%20in%20the%20Entropy%20Code%20Domain) [IH & MMSec 2018, Best Paper Award]
**[[paper (ACM)](https://dl.acm.org/citation.cfm?id=3206011)]**
**[[paper (pdf)](http://www.media-security.net/?p=809)]**
**[[Others](http://www.media-security.net/?p=809)]**
**[[DataSet](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper)]**

## Necessary Package
tensorflow-gpu==1.4 or later, numpy, pandas, matplotlib, scikit-image, scikit-learn, librosa (depend on **FFmpeg**)

You can use command **pip install -r requirements.txt** to install all packages mentioned above. If you don't want to change your version of tensorflow, you can use **virtualenv** to create a new python run environment.

## How to use
1. install [**python3.x**](https://www.python.org/) or [**Anaconda**](https://repo.continuum.io/archive/) and add the path into the environment variable
2. **GPU** run environment [configure](https://blog.csdn.net/yhaolpz/article/details/71375762?locationNum=14&fps=1) if train the network (optional)
3. install all dependent packages mentioned above (open **setup/requirements.txt** and input "**pip install -r requirements**" into your cmd window)
4. **run** the code as the [**example**](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/config_file) as follows
5. use **tensorboard** to visualize the train process such as the accuracy and loss curve of train and valid. The command is "**tensorboard --logdir=/path/to/log-directory**"

## File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
01      |   audio_preprocess.py     |   include some pre-process methods for **audio**
02      |   text_preprocess.py      |   include some pre-process methods for **text**
03      |   image_preprocess.py     |   include some pre-process methods for **image** 
04      |   file_preprocess.py      |   get the name, size and type of the **file**
05      |   samples_make.py         |   samples make script (cover, MP3Stego_cover, MP3stego, HCM, EECS)
06      |   config.py               |   all configuration for the system running
07      |   filters.py              |   some **filters** used for pre-processing such as kv kernel or other **rich model**
08      |   **main.py**             |   the main program
09      |   manager.py              |   **GPU** management (free GPU selection **automatically**)
10      |   layer.py                |   basic unit in CNN such as **conv layer**, **pooling layer**, **BN layer** and so on
11      |   network                 |   various networks scirpt for audio, image steganalysis and other image classification task including **VGG19**, **LeNet** and **ourselves' network**
12      |   utils.py                |   some useful tools such as **minibatch**, **get_model_info**, 
13      |   run.py                  |   the **train** and **test** of the network **get_weights**, **get_biases** and so on
14      |   dataset.py              |   some functions of tfrecord read and write
15      |   lstm.py                 |   lstm network which uesd for steganalysis
16      |   test_script.py          |   a script for function test
17      |   setup                   |   a requirements.txt in this folder, which is used to install all packages in this system
18      |   tools                   |   some useful tools in this folder, which are used to QMDCT coefficients extraction and others
19      |   data_processing         |   the scripts in this folder are used to make dataset
20      |   config_file             |   three files, config_train, config_test and config_steganalysis, in this folder are uesd to send the paramters into the network, like the usage in Caffe

## Reference
**[1]** **tensorflow** API: https://www.tensorflow.org/ <br>
**[2]** **tensorlayer** API: http://tensorlayer.readthedocs.io/en/latest/ <br>
**[3]** **tensorboard** usage: http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/graph_viz.html <br>
**[4]** **FFmpeg**: http://www.ffmpeg.org/download.html <br>
**[5]** **Python**: https://www.python.org/ <br>
**[6]** **Anaconda**: https://repo.continuum.io/archive/ <br>
**[7]** **librosa API**: http://librosa.github.io/librosa/core.html <br>
**[8]** **librosa introduction (Chinese)**: https://www.cnblogs.com/xingshansi/p/6816308.html <br>

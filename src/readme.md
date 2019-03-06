## File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
01      |   config_file             |   three files, config_train, config_test and config_steganalysis, in this folder are uesd to send the paramters into the network, like the usage in Caffe
02      |   HPFs                    |   including all **high pass filters** and filters.py
03      |   matlab_scripts          |   matlab scripts for jpeg image read and write
04      |   modules                 |   basic modules of arbitrary size, densenet, resnet and inception
05      |   networks                |   all designed networks are contained in this folder, audio and image steganalysis, classification
06      |   audio_preprocess.py     |   include some pre-process methods for **audio**
07      |   config.py               |   all configuration and parameters setting for the system running
08      |   dataset.py              |   some functions of tfrecord read and write
09      |   distribution.py         |   distribution calculation
10      |   file_preprocess.py      |   include some pre-process methods for **file**
11      |   image_preprocess.py     |   include some pre-process methods for **image**
12      |   layer.py                |   basic unit in CNN such as **conv layer**, **pooling layer**, **BN layer** and so on
13      |   **main.py**             |   the main program
14      |   manager.py              |   **GPU** management (free GPU selection **automatically**)
15      |   run.py                  |   the **train** and **test** of the network **get_weights**, **get_biases** and so on
16      |   text_preprocess.py      |   include some pre-process methods for **text**
17      |   utils.py                |   some useful tools such as **minibatch**, **get_data_batch** and so on

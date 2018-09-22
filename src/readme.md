## File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
01      |   audio_preprocess.py     |   include some pre-process methods for **audio**
02      |   text_preprocess.py      |   include some pre-process methods for **text**
03      |   image_preprocess.py     |   include some pre-process methods for **image**
04      |   distribution.py         |   distribution calculation
05      |   config.py               |   all configuration and parameters setting for the system running
06      |   filters.py              |   some **filters** used for pre-processing such as kv kernel or other **rich model**
07      |   **main.py**             |   the main program
08      |   manager.py              |   **GPU** management (free GPU selection **automatically**)
09      |   dataset.py              |   tfrecord read and write
10      |   layer.py                |   basic unit in CNN such as **conv layer**, **pooling layer**, **BN layer** and so on
11      |   utils.py                |   some useful tools such as **minibatch**, **get_data_batch**, 
12      |   run.py                  |   the **train** and **test** of the network **get_weights**, **get_biases** and so on
13      |   dataset.py              |   some functions of tfrecord read and write
14      |   lstm.py                 |   lstm network which uesd for steganalysis
15      |   networks                |   all designed networks are contained in this folder, audio and image steganalysis, classification
16      |   config_file             |   three files, config_train, config_test and config_steganalysis, in this folder are uesd to send the paramters into the network, like the usage in Caffe

## Usage
If you use json mode, it's OK to open the json file and modify the corresponding parameters. And, if you use command line mode, modify the corresponding parameters and copy/paste it.

### Json mode
    Example(train):
    (sudo) python main.py (train)
 
 ---
    Example(test):
    (sudo) python main.py test

---
    Example(steganalysis):
    (sudo) python main.py steganalysis

### Command line mode

    Example(train and validation):
    sudo python main.py \
    --task_name xxx\
    --mode train \
    --network wasdn \
    --carrier qmdct \
    --gpu_selection manu \
    --gpu 0 \
    --height 200 \
    --width 380 \
    --channel 1 \
    --batch_size 64 \
    --cover_train_path xxx \
    --cover_valid_path xxx \
    --stego_train_path xxx \
    --stego_valid_path xxx \
    --tfrecords_path xxx \
    --models_path xxx \
    --logs_path xxx \
    --task_name xxx \
---
    Example(test): 
    sudo python main.py \
    --mode test \
    --network wasdn \
    --carrier qmdct \
    --gpu_selection manu \
    --gpu 0 \
    --height 200 \
    --width 380 \
    --channel 1 \
    --batch_size 64 \
    --cover_test_path xxx \
    --stego_test_path xxx \
    --models_path xxx \
---
    Example(steganalysis for **one**):
    sudo python main.py \
    --mode steganalysis \
    --submode one \
    --network wasdn \
    --carrier qmdct \
    --gpu_selection manu \
    --gpu 0 \
    --height 200 \
    --width 380 \
    --channel 1 \
    --steganalysis_file_path xxx \
    --models_path xxx \
---
    Example(steganalysis for **batch**):
    sudo python main.py \
    --mode steganalysis \
    --submode batch \
    --network wasdn \
    --carrier qmdct \
    --gpu_selection manu \
    --gpu 0 \
    --height 200 \
    --width 380 \
    --channel 1 \
    --steganalysis_files_path xxx \
    --models_path xxx \

## Path Mode
There are three modes for dataset path setting: **full**, **simple** and **semi**. The three modes are independent each other, and only one setting is needed at a time.

### Full
cover_train_path -- path of cover train files <br>
cover_valid_path -- path of cover validation files <br>
stego_train_path -- path of stego train files <br>
stego_valid_path -- path of stego validation files <br>

### Simple
cover_train_path -- path of cover train files <br>
cover_valid_path -- path of cover validation files <br>
stego_train_path -- path of stego train files <br>
stego_valid_path -- path of stego validation files <br>

### semi
cover_files_path -- path of cover files <br>
stego_files_path -- path of stego files <br>

## Parameters
All parameters match the mode of **argparse**, and commonly used parameters are **bold**.

### training and validation

Parameters                  | Function                                                          | Option
:-                          | :-                                                                | :-
 **path_mode**              | mode of path                                                      | simple(default) and full
 **task_name**              | name of task                                                      | task name is the same with the steg folder name if you select simple path mode
 **cover_train_path**       | path of cover train files for full path mode                      | -
 **cover_valid_path**       | path of cover validation files for full path mode                 | -
 **stego_train_path**       | path of stego train files for full path mode                      | -
 **stego_valid_path**       | path of stego validation files for full path mode                 | -
 **cover_files_path**       | directory of cover files for simple path mode                     | -
 **stego_files_path**       | directory of stego files for simple path mode                     | -
 **cover_files_path**       | path of cover files for semi path mode                            | -
 **stego_files_path**       | path of stego files for semi path mode                            | -
 **tfrecords_path**         | path of tfrecords                                                 | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 **logs_path**              | path of log files including train and validation                  | -
 gpu_selection              | mode of gpu selection                                             | auto(default), manu
 **mode**                   | mode of running                                                   | train(default), test and steganalysis
 **carrier**                | carrier for steganalysis                                          | qmdct(default), audio and image
 **network**                | name of designed network                                          | -
 **batch_size**             | batch size for training                                           | 16(default)
 **learning_rate**          | initialized learning rate                                         | 1e-3(default)
 **epoch**                  | epoch of training                                                 | 500(default)
 seed                       | seed of random generation                                         | 1(default)
 **is_regulation**          | whether add regulation or not                                     | True(default), False
 coeff_regulation           | coefficient of regulation                                         | 1e-3(default)
 loss_method                | method of loss function                                           | sigmoid_cross_entropy, softmax_cross_entropy, sparse_softmax_cross_entropy(default)
 class_num                  | number of classifier                                              | 2(default)
 **height**                 | height of input data matrix                                       | -
 **width**                  | width of input data matrix                                        | -
 **channel**                | channel of input data matrix                                      | 1(default), 3
 decay_method               | method of learning rate decay                                     | fixed, step, exponential(defaut), inverse_time, natural_exp, polynomial
 decay_step                 | step for learning rate decay                                      | 5000(default)
 decay_rate                 | rate of learning rate decay                                       | 0.9(default)
 staircase                  | whether the decay the learning rate at discrete intervals or not  | True and False(default)
 max_to_keep                | number of models to be saved                                      | 3(default)

### test

Parameters                  | Function                                                          | Option
:-                          | :-                                                                | :-
 **cover_test_path**        | path of cover train files                                         | -
 **stego_test_path**        | path of stego test files                                          | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 gpu_selection              | mode of gpu selection                                             | auto(default), manu
 **mode**                   | mode of running                                                   | train, test(default) and steganalysis
 **carrier**                | carrier for steganalysis                                          | qmdct(default), audio and image
 **network**                | name of designed network                                          | -
 class_num                  | number of classifier                                              | 2(default)
 **height**                 | height of input data matrix                                       | -
 **width**                  | width of input data matrix                                        | -
 **channel**                | channel of input data matrix                                      | 1(default), 3

### steganalysis

Parameters                  | Function                                                          | Option
:-                          | :-                                                                | :-
 **steganalysis_file_path** | path of file for steganalysis                                     | -
 **steganalysis_files_path**| path of files for steganalysis                                    | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 gpu_selection              | mode of gpu selection                                             | auto(default), manual
 **mode**                   | mode of running                                                   | train, test and steganalysis(default)
 **submode**                | submode of steganalysis                                           | one and batch(default, this parameter is just for the mode of steganalysis)
 **carrier**                | carrier for steganalysis                                          | qmdct(default), audio and image
 **network**                | name of designed network                                          | -
 class_num                  | number of classifier                                              | 2(default)
 **height**                 | height of input data matrix                                       | -
 **width**                  | width of input data matrix                                        | -
 **channel**                | channel of input data matrix                                      | 1(default), 3

## Usage
### json mode
    Example(train):
    sudo python main.py (train)
 
 ---
    Example(test):
    sudo python main.py test

---
    Example(steganalysis):
    sudo python main.py steganalysis

### command line mode

    Example(train):
    sudo python main.py \
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
    --tfreocrds_path xxx \
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
    Example(steganalysis for one): 
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
    Example(steganalysis for batch): 
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

## Parameters
All parameters match the mode of **argparse**, and commonly used parameters are **bold**.

### training and validation

Param                       | Function                          | option
:-                          | :-                                | :-
 **cover_train_path**       | path of cover train files         | -
 **cover_valid_path**       | path of cover validation files    | -
 **stego_train_path**       | path of stego train files         | -
 **stego_valid_path**       | path of stego validation files    | -
 **tfrecords_path**         | path of tfreocrds                 | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 **logs_path**              | path of log files including train and validation                  | -
 gpu_selection              | mode of gpu selection             | auto(default), manu
 **mode**                   | mode of running                   | train(default), test and steganalysis
 **carrier**                | carrier for steganalysis          | qmdct(default), audio and image
 **network**                | name of designed network          | -
 **task_name**              | name of task                      | -
 **path_mode**              | mode of path                      | simple(default) and full
 **batch_size**             | batch size for training           | 128(default)
 **learning_rate**          | initialized learing rate          | 1e-3(default)
 **epoch**                  | epoch of training                 | 500(default)
 seed                       | seed of random generation         | 1(default)
 **is_regulation**          | whether add regulation or not     | True(default), False
 coeff_regulation           | coefficient of regulation         | 1e-3(default)
 loss_method                | method of loss funcion            | sigmoid_cross_entropy, softmax_cross_entropy, sparse_softmax_cross_entropy(default)
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3
 decay_method               | method of learning rate decay     | fixed, step, exponential(defaut), inverse_time, natural_exp, polynomial
 decay_step                 | step for learning rate decay      | 5000(default)
 decay_rate                 | rate of learing rate decay        | 0.9(default)
 staircase                  | whether the decay the learning rate at discrete intervals or not | True and False(default)
 max_to_keep                | number of models to be saved      | 3(defalut)

### test

Param                       | Function                          | option
:-                          | :-                                | :-
 **cover_test_path**        | path of cover train files         | -
 **stego_test_path**        | path of stego test files          | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 gpu_selection              | mode of gpu selection             | auto(default), manu
 **mode**                   | mode of running                   | train, test(default) and steganalysis
 **carrier**                | carrier for steganalysis          | qmdct(default), audio and image
 **network**                | name of designed network          | -
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3

### steganalysis

Param                       | Function                          | option
:-                          | :-                                | :-
 **steganalysis_file_path** | path of file for steganalysis     | -
 **steganalysis_files_path**| path of files for steganalysis    | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 gpu_selection              | mode of gpu selection             | auto(default), manual
 **mode**                   | mode of running                   | train, test and steganalysis(default)
 **submode**                | submode of steganalysis           | one and batch(default, this parameter is just for the mode of steganalysis)
 **carrier**                | carrier for steganalysis          | qmdct(default), audio and image
 **network**                | name of designed network          | -
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3

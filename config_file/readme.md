# Parameters
All parameters match the mode of **argparse**, and commonly used parameters are **bold**.
 
## training and validation

Param                       | Function                          | option
:-                          | :-                                | :-
 **cover_train_path**       | path of cover train files         | -
 **cover_valid_path**       | path of cover validation files    | -
 **stego_train_path**       | path of stego train files         | -
 **stego_valid_path**       | path of stego validation files    | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 **logs_path**              | path of log files including train and validation                  | -
 gpu_selection              | mode of gpu selection             | auto(default), manu
 **mode**                   | mode of running                   | train(default), test and steganalysis
 **carrier**                | carrier for steganalysis          | qmdct(default), audio and image
 **network**                | name of designed network          | -
 **batch_size**             | batch size for training           | 128(default)
  **learing_rate**          | initialized learing rate          | 1e-3(default)
 **epoch**                  | epoch of training                 | 500(default)
 seed                       | seed of random generation         | 1(defalut)
 **is_regulation**          | whether add regulation or not     | True(default), false
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

## test

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

## steganalysis

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
 
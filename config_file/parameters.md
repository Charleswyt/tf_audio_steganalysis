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
 gpu_selection              | mode of gpu selection             | auto(default), manual
 **mode**                   | mode of running                   | train(default), test and steganalysis
 **submode**                | submode of steganalysis           | one and batch(default)
 **carrier**                | carrier for steganalysis          | audio(default) and image
 **network**                | name of designed network          | -
 **batch_size_train**       | batch size for training           | 64(default)
 **batch_size_validation**  | batch size for validation         | 16(default)
  **learing_rate**          | initialized learing rate          | 1e-3(default)
 **epoch**                  | epoch of training                 | 500(default)
 seed                       | seed of random generation         | 1(defalut)
 **is_regulation**          | whether add regulation or not     | True(fault), false
 coeff_regulation           | coefficient of regulation         | 1e-3(default)
 loss_method                | method of loss funcion            | sigmoid_cross_entropy, softmax_cross_entropy, sparse_softmax_cross_entropy(default)
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3
 is_abs                     | whether make absolute or not      | True and False(default)
 is_trunc                   | whether make truncation or not    | True and False(default)
 is_diff                    | whether make difference or not    | True and False(default)
 is_diff_abs                | whether make diff and abs or not  | True and False(default)
 is_abs_diff                | whether make abs and diff or not  | True and False(default)
 threshold                  | threshold of truncation           | True and False(default)
 order                      | order of difference               | True and False(default)
 direction                  | direction of diff(0-row, 1-col)   | True and False(default)
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
 gpu_selection              | mode of gpu selection             | auto(default), manual
 **mode**                   | mode of running                   | train(default), test and steganalysis
 **submode**                | submode of steganalysis           | one and batch(default)
 **carrier**                | carrier for steganalysis          | audio(default) and image
 **network**                | name of designed network          | -
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3
 is_abs                     | whether make absolute or not      | True and False(default)
 is_trunc                   | whether make truncation or not    | True and False(default)
 is_diff                    | whether make difference or not    | True and False(default)
 is_diff_abs                | whether make diff and abs or not  | True and False(default)
 is_abs_diff                | whether make abs and diff or not  | True and False(default)
 threshold                  | threshold of truncation           | True and False(default)
 order                      | order of difference               | True and False(default)
 direction                  | direction of diff(0-row, 1-col)   | True and False(default)

## steganalysis

Param                       | Function                          | option
:-                          | :-                                | :-
 **files_path**             | path of files for steganalysis    | -
 **models_path**            | path of models including checkpoint, data, index and meta files   | -
 gpu_selection              | mode of gpu selection             | auto(default), manual
 **mode**                   | mode of running                   | train(default), test and steganalysis
 **submode**                | submode of steganalysis           | one and batch(default)
 **carrier**                | carrier for steganalysis          | audio(default) and image
 **network**                | name of designed network          | -
 class_num                  | number of classifier              | 2(default)
 **height**                 | height of input data matrix       | -
 **width**                  | width of input data matrix        | -
 **channel**                | channel of input data matrix      | 1(default), 3
 is_abs                     | whether make absolute or not      | True and False(default)
 is_trunc                   | whether make truncation or not    | True and False(default)
 is_diff                    | whether make difference or not    | True and False(default)
 is_diff_abs                | whether make diff and abs or not  | True and False(default)
 is_abs_diff                | whether make abs and diff or not  | True and False(default)
 threshold                  | threshold of truncation           | True and False(default)
 order                      | order of difference               | True and False(default)
 direction                  | direction of diff(0-row, 1-col)   | True and False(default)
 
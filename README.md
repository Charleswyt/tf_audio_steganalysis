# Audio Steganalysis with CNN
@ author: Wang Yuntao <br>
## Necessary Package
tensorflow-gpu==1.3, numpy, matplotlib

## CNN Architecture

## Steganography Algorithm
HCM(Huffman Code Mapping), EECS(Equal Length Entropy Codes Substitution)

## Results

## Run
* install python3.5 and add the path into environment variable
* pip install tensorflow==1.3 numpy scikit-image
* run the code as the example as follows

## Command Parser
Command: python3 main.py --argument 1 --argument 2 ... --argument N <br>

    Namespace(batch_size=128, batch_size_train=128, batch_size_valid=128, bitrate=128, block=2, carrier='audio', cover_train_dir=None, cover_valid_dir=None, data_dir=None, decay_method='exponential', decay_rate=0.9, decay_step=5000, direction=0, downsampling=False, end_index_train=-1, end_index_valid=-1, epoch=500, height=512, is_abs=False, is_diff=False, is_regulation=True, is_trunc=False, keep_checkpoint_every_n_hours=0.5, learning_rate=0.001, log_dir=None, logs_path='/home/zhanghong/code/CatKing/steganalysis_CNN/logs', max_to_keep=3, mode='test', model_dir='/home/zhanghong/code/CatKing/steganalysis_CNN/models/stegshi', model_file_name='audio_steganalysis', model_file_path=None, models_path='/home/zhanghong/code/CatKing/steganalysis_CNN/models', network='stegshi', order=2, relative_payload='2', seed=1, staircase=False, start_index_train=0, start_index_valid=0, stego_method='EECS', stego_train_dir=None, stego_valid_dir=None, submode='one', test_file_path='/home/zhanghong/data/image/val/512_stego/result_7518.pgm', test_files_dir=None, threshold=15, width=512)


Example: <br>
    
    --train
    
    simple 1: (sudo) python3 main.py --mode train --data\_dir /home/"home_name"/data --height 200 --width 380
    
    simple 2: (sudo) python3.5 main.py --mode train --carrier image --height 512 --width 512 --network stegshi --batch_size 64 --end_index_train 6000 --end_index_valid 1500 --cover_train_dir /home/zhanghong/data/image/train/512_cover --cover_valid_dir /home/zhanghong/data/image/val/512_cover/ --stego_train_dir /home/zhanghong/data/image/train/512_stego/ --stego_valid_dir /home/zhanghong/data/image/val/512_stego/ --logs_path /home/zhanghong/code/CatKing/steganalysis_CNN/logs/stegshi --models_path /home/zhanghong/code/CatKing/steganalysis_CNN/models/stegshi

    --test
    
    sample 1: (sudo) python3.5 main.py --mode test --submode one --network stegshi --model_dir /home/zhanghong/code/CatKing/steganalysis_CNN/models/stegshi --file_path /home/zhanghong/data/image/test/12138.pgm
    
    sample 2: (sudo) python3.5 main.py --mode test --submode one --network stegshi --height 512 --width 512 --model_dir /home/zhanghong/code/CatKing/steganalysis_CNN/models/stegshi --test_file_path /home/zhanghong/data/image/val/512_cover/7501.pgm
    
    sample 3: (sudo) python3.5 main.py --mode test --submode batch --network stegshi --model_dir /home/zhanghong/code/CatKing/steganalysis_CNN/models/stegshi --test_file_path /home/zhanghong/data/image/test/ --label_file_path /home/zhanghong/data/image/test/
    
    
**arguments:** <br>
**Mode:** mode, test, data\_dir <br>
**Data\_info:** bitrate, relative\_payload, stego\_method, model\_dir, log\_dir <br>
**Hyper\_paramsters:** batch\_size, batch\_size\_train, batch_size_valid, epoch, learning\_rate, gpu, seed, decay\_step, decay\_rate, staircase <br> 
**Path:** cover\_train\_dir, cover\_valid\_dir, stego\_train\_dir, stego\_valid\_dir, models\_path, logs\_path <br>
**Network:** network <br>

**illustration:** <br>
**mode:** train | test <br>
**test:** one | batch <br>
**data_dir:** data dir <br>
**bitrate:** 128 | 192 | 256 | 320 <br>
**stego_method:** EECS | HCM-Gao | HCM-Yan <br>
**relative\_payload:**

    2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10, if the steganography algorithm is the EECS algorithm
    1 | 3 | 5, otherwise
The introdction of each network


    network1  : The proposed network (最终选定的网络)
    network1_1: Remove the BN layer (去掉BN层)
    network1_2: Average pooling layer is used for subsampling (将所有的降采样方式改为平均池化方式)
    network1_3: Convolutional layer with stride 2 is used for subsampling (将所有的降采样方式改为卷积池化方式)
    network1_4: Replace the convolutional kernel with 5x5 kernel (将卷积核尺寸由3 x 3改为5 x 5)
    network1_5: ReLu is used as the activation function (将激活函数由Tanh改为ReLu)
    network1_6: Leaky-ReLu is used as the activation function (将激活函数由tanh改为Leaky-ReLu)
    network1_7: Deepen the network to block convolution layers (加深网络)
    network1_8: Design a network to steganalyze audios of arbitrary size (解决可变尺寸输入数据的训练问题)
    stegshi   : Xu-Net which is used for image steganalysis
    
    Note: HPF and ABS is applied at the pre-processing
    
* The method of pre-processing
    There are positive and negative values in QMDCT coefficients matrix. The values in interval [-15, 15] is modified.
    The ratio of values in [-15, 15] is more than 99%, as the figure shown.
    * Abs
    * Truncation
    * Down-sampling

* Training <br>
    `python main.py train -data_dir EECS 128 2 -model_dir -log_dir`

* Test <br>
    `python main.py test -data_dir EECS 128 2 -model_dir`

* Data Folder Name <br>

    
    cover

        128 | 192 | 256 | 320
    stego

        EECS
            128_W_2_H_7_ER_10, 128_W_3_H_7_ER_10, ..., 128_W_10_H_7_ER_10
            192_W_2_H_7_ER_10, 192_W_3_H_7_ER_10, ..., 192_W_10_H_7_ER_10
            256_W_2_H_7_ER_10, 256_W_3_H_7_ER_10, ..., 256_W_10_H_7_ER_10
            320_W_2_H_7_ER_10, 320_W_3_H_7_ER_10, ..., 320_W_10_H_7_ER_10
        W: The width of the parity-check matrix, W = 1 / α, and α is the relative payload
        H: the height of the parity-check matrix
        ER: The number of fremes used for embedding = The number of whole frames * ER
        
        HCM-Gao
            128_01, 128_03, 128_05, 128_10
            192_01, 192_03, 192_05, 192_10
            256_01, 256_03, 256_05, 256_10
            320_01, 320_03, 320_05, 320_10
        01, 03, 05, 10 is the ER as shown above
        
        HCM-Yan
            128_01, 128_03, 128_05, 128_10
            192_01, 192_03, 192_05, 192_10
            256_01, 256_03, 256_05, 256_10
            320_01, 320_03, 320_05, 320_10

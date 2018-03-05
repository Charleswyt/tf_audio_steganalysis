# Audio Steganalysis with CNN
## Necessary Package
tensorflow-gpu==1.3, numpy, matplotlib

## CNN Architecture

## Steganography Algorithm
HCM(Huffman Code Mapping), EECS(Equal Length Entropy Codes Substitution)

## Results

## Run
Command: python3 main.py --argument 1 --argument 2 ... --argument N <br>
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

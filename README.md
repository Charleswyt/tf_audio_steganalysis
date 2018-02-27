# Audio Steganalysis with CNN
## Necessary Package
tensorflow-gpu==1.3, numpy, matplotlib

## CNN Architecture

## Steganography Algorithm
HCM(Huffman Code Mapping), EECS(Equal Length Entropy Codes Substitution)

## Results

## Run
Command: python3 main.py <mode> <data\_dir> <stego\_method> \<bitrate> <relative\_payload> <model\_dir> <log\_dir> <br>
data_dir: data dir <br>
mode: train | test <br>
stego_method: EECS | HCM-Gao | HCM-Yan <br>
bitrate: 128 | 192 | 256 | 320 <br>
relative\_payload:

    2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10, if the steganography algorithm is the EECS algorithm
    0.1 | 0.3 | 0.5, otherwise
                   

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
        
        HCM-Gao
            128_01, 128_03, 128_05, 128_10
            192_01, 192_03, 192_05, 192_10
            256_01, 256_03, 256_05, 256_10
            320_01, 320_03, 320_05, 320_10
        
        HCM-Yan
            128_01, 128_03, 128_05, 128_10
            192_01, 192_03, 192_05, 192_10
            256_01, 256_03, 256_05, 256_10
            320_01, 320_03, 320_05, 320_10


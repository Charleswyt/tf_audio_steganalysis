## RHFCN: Fully CNN-based Steganalysis of MP3 with Rich High-Pass Filtering

## Structure of RHFCN
![Structure of RHFCN](https://i.imgur.com/X3Pp0Sm.jpg)
### Structure of Conv Block
![Structure of conv block](https://i.imgur.com/9fJe1de.jpg)
## Abstract
Recent studies have shown that convolutional neural networks (CNNs) can boost the performance of audio steganalysis. In this paper, we propose a well-designed fully CNN architecture for MP3 steganalysis based on rich high-pass filtering (HPF). **On the one hand**, multi-type HPFs are employed for “residual” extraction to enlarge the traces of the signal in view of the truth that signal introduced by secret messages can be seen as high-pass frequency noise. **On the other hand**, to utilize the spatial characteristics of feature maps better, fully connected (Fc) layers are replaced with convolutional layers. **Moreover**, this fully CNN architecture can be applied to the steganalysis of MP3 with size mismatch. The proposed network is evaluated on various MP3 steganographic algorithms, bitrates and relative payloads, and the experimental results demonstrate that our proposed network performs better than state-of-the-art methods.

## How to Cite
    @inproceedings{WangYZS19,
    author    = {Yuntao Wang and Xiaowei Yi and Xianfeng Zhao and Ante Su},
    title     = {{RHFCN: Fully CNN-based Steganalysis of MP3 with Rich High-Pass Filtering}},
    booktitle = {Proceedings of the 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2019)},
    pages     = {2627--2631},
    year      = {2019},
	address   = {Brighton, UK},
	publisher = {IEEE}
    }

## The description of each network
        RHFCN   : The proposed network for MP3 steganalysis
        RHFCN1_1: Remove rich HPF module
        RHFCN1_2: Quit removing Fc layers
        RHFCN1_3: Remove rich HPF module and quit removing Fc layers
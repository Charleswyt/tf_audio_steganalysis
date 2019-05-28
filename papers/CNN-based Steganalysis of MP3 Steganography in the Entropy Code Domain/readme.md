## WASDN: Wang Audio Steganalysis Deep Network
## Structure of WASDN
![Structure of WASDN](https://i.imgur.com/h0o5lfB.jpg)

## Abstract
This paper presents an effective steganalytic scheme based on CNN for detecting MP3 steganography in the entropy code domain. These steganographic methods hide secret messages into the compressed audio stream through Huffman code substitution, which usually achieve high capacity, good security and low computational complexity. **First**, unlike most previous CNN based steganalytic methods, the quantified modified DCT (QMDCT) coefficients matrix is selected as the input data of the proposed network. **Second**, a high pass filter is used to extract the residual signal, and suppress the content itself, so that the network is more sensitive to the subtle alteration introduced by the data hiding methods. **Third**, the 1 x 1 convolutional kernel and the batch normalization layer are applied to decrease the danger of overfitting and accelerate the convergence of the back-propagation. **In addition**, the performance of the network is optimized via fine-tuning the architecture. The experiments demonstrate that the proposed CNN performs far better than the traditional handcrafted features. In particular, the network has a good performance for the detection of an **adaptive** MP3 steganography algorithm, equal length entropy codes substitution (**EECS**) algorithm which is hard to detect through conventional handcrafted features. The network can be applied to various bitrates and relative payloads seamlessly. **Last but not the least**, a sliding window method is proposed to steganalyze audios of arbitrary size.

## How to Cite
    @inproceedings{WangYYZX18,
    author    = {Yuntao Wang and Kun Yang and Xiaowei Yi and Xianfeng Zhao and Zhoujun Xu},
    title     = {{CNN-based Steganalysis of {MP3} Steganography in the Entropy Code Domain}},
    booktitle = {Proceedings of the 6th {ACM} Workshop on Information Hiding and Multimedia Security (IH&MMSec 2018)},
    pages     = {55--65},
    year      = {2018},
	address   = {Innsbruck, Austria},
	publisher = {ACM}
    }

## Description of Each Variant
        WASDN   : The proposed network for MP3 steganalysis (Wang Audio Steganalysis Deep Network)
        WASDN1_1: Remove the BN layer
        WASDN1_2: Average pooling layer is used for subsampling
        WASDN1_3: Convolutional layer with stride 2 is used for subsampling
        WASDN1_4: Replace the convolutional kernel with 5x5 kernel
        WASDN1_5: ReLu is used as the activation function
        WASDN1_6: Leaky-ReLu is used as the activation function
        WASDN1_7: Deepen the network to block convolutional layers
        WASDN1_8: Remove the 1x1 convolutional layers
        WASDN1_9: Remove the HPF layer
        WASDN2_1: Remove the first BN layer in the first group
        WASDN2_2: Remove the first BN layers in the first two groups
        WASDN2_3: Remove the first BN layers in the first four groups
        WASDN2_4: Add BN layers at the top of 3x3 conv layer
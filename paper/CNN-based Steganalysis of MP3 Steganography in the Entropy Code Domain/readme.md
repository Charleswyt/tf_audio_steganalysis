## The architecture of the network (Wang Audio Steganalysis Deep Network, WASDN)
![The structure of the proposed network](https://i.imgur.com/h0o5lfB.jpg)

## Abstract
This paper presents an effective steganalytic scheme based on CNN for detecting MP3 steganography in the entropy code domain. These steganographic methods hide secret messages into the compressed audio stream through Huffman code substitution, which usually achieve high capacity, good security and low computational complexity. First, unlike most previous CNN based steganalytic methods, the quantified modified DCT (QMDCT) coefficients matrix is selected as the input data of the proposed network. Second, a high pass filter is used to extract the residual signal, and suppress the content itself, so that the network is more sensitive to the subtle alteration introduced by the data hiding methods. Third, the 1 x 1 convolutional kernel and the batch normalization layer are applied to decrease the danger of overfitting and accelerate the convergence of the back-propagation. In addition, the performance of the network is optimized via fine-tuning the architecture. The experiments demonstrate that the proposed CNN performs far better than the traditional handcrafted features. In particular, the network has a good performance for the detection of an adaptive MP3 steganography algorithm, equal length entropy codes substitution (EECS) algorithm which is hard to detect through conventional handcrafted features. The network can be applied to various bitrates and relative payloads seamlessly. Last but not the least, a sliding window method is proposed to steganalyze audios of arbitrary size.

## Cite it
    @inproceedings{DBLP:conf/ih/WangYYZX18,
    author    = {Yuntao Wang and
                Kun Yang and
                Xiaowei Yi and
                Xianfeng Zhao and
                Zhoujun Xu},
    title     = {CNN-based Steganalysis of {MP3} Steganography in the Entropy Code
                Domain},
    booktitle = {Proceedings of the 6th {ACM} Workshop on Information Hiding and Multimedia
                Security, Innsbruck, Austria, June 20-22, 2018},
    pages     = {55--65},
    year      = {2018},
    crossref  = {DBLP:conf/ih/2018},
    url       = {http://doi.acm.org/10.1145/3206004.3206011},
    doi       = {10.1145/3206004.3206011},
    timestamp = {Thu, 21 Jun 2018 08:37:36 +0200},
    biburl    = {https://dblp.org/rec/bib/conf/ih/WangYYZX18},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }

## The description of each network
**network for _audio_ steganalysis**

        network1_1 : Remove the BN layer
        network1_2 : Average pooling layer is used for subsampling
        network1_3 : Convolutional layer with stride 2 is used for subsampling
        network1_4 : Replace the convolutional kernel with 5x5 kernel
        network1_5 : ReLu is used as the activation function
        network1_6 : Leaky-ReLu is used as the activation function
        network1_7 : Deepen the network to block convolutional layers
        network1_8 : Remove the 1x1 convolutional layers
        network1_9 : Remove the HPF layer
        network1__1: Remove the first BN layer in the first group
        network1__2: Remove the first BN layers in the first two groups
        network1__3: Remove the first BN layers in the first four groups
        network1__4: Add BN layers at the top of 3x3 conv layer
## The architecture of the network (Wang Audio Steganalysis Deep Network, WASDN)
![The structure of the proposed network](https://i.imgur.com/h0o5lfB.jpg)

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
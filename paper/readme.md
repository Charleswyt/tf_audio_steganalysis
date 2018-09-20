## Steganographic Algorithm
**HCM** (Huffman Code Mapping) and **EECS** (Equal Length Entropy Codes Substitution, an adaptive MP3 steganographic algorithm with [**STC**](http://dde.binghamton.edu/download/syndrome/) and **distortion function** based on **psychological acoustics model** (**PAM**)).

Note: All **built-in** MP3 algorithms embeds secret messages in the process of MP3 encoding, which will change QMDCT coefficients of MP3. So, this network can be applied to detect all data hiding methods which have impact on the QMDCT coefficients.

All steganographic algorithms are coded by Kun Yang (E-Mail: yangkun9076@iie.ac.cn)
## Dataset
The dataset can be downloaded from [**Audio Steganalysis Dataset, IIE (ASDIIE)**](https://pan.baidu.com/s/1rYCzJRksHkgbOOYI9MqQjA) <br>
The *extraction password* is "**z28d**".


## The method of pre-processing
    
There are positive and negative values in QMDCT coefficients matrix. The values in interval **[-15, 15]** is modified.
The ratio of values in **[-15, 15]** is more than **99%**, as the figure shown. <br>

* Abs <br>
* Truncation <br>
* Down-sampling <br>
![The distribution of QMDCT coefficients](https://i.imgur.com/vDJ2gWm.jpg)
The proportion of zero value is far greater than others, so we remove the zero value in the right figure for better visual effects.

## Reference
**[1]** Haiying Gao. 2007. **The MP3 steganography algorithm based on Huffman coding**. Acta Scientiarum Naturalium Universitatis Sunyatseni 4 (2007), 009. <br>
**[2]** Diqun Yan, Rangding Wang, and Liguang ZHANG. 2011. **A high capacity MP3 steganography based on Huffman coding**. Journal of Sichuan University (Natural Science Edition) 6 (2011), 013. <br>
**[3]** Kun Yang, Xiaowei Yi, Xianfeng Zhao, and Linna Zhou. 2017. **Adaptive MP3 Steganography Using Equal Length Entropy Codes Substitution**. In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 202–216. <br>
**[4]** Yanzhen Ren, Qiaochu Xiong, and Lina Wang. 2017. **A Steganalysis Scheme for AAC Audio Based on MDCT Difference Between Intra and Inter Frame**. In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 217–231. <br>
**[5]** Chao Jin, Rangding Wang, and Diqun Yan. 2017. **Steganalysis of MP3Stego with low embedding-rate using Markov feature**. Multimedia Tools and Applications 76, 5 (2017), 6143–6158. <br>
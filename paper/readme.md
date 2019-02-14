## Dataset
The dataset can be downloaded from [**Audio Steganalysis Dataset, IIE (ASDIIE)**](https://pan.baidu.com/s/1rYCzJRksHkgbOOYI9MqQjA)

The *extraction password* is "**z28d**".

You can also get all data via [**IEEEDataPort**](http://ieee-dataport.org/documents/audio-steganalysis-dataset) (Building...).

We divide our database into vocal songs and light music, and we mainly steganalyze **vocal songs**.

Besides, 16,000 [**BBC Sound Effects**](http://bbcsfx.acropolis.org.uk/) are made available by the **BBC** in **WAV** format to download for use under the terms of the RemArc Licence.

There are other [**audio databases**](https://github.com/Charleswyt/tf_audio_steganalysis/blob/master/paper/audio_database.md) we can use.

According to our experiments, **10000** cover-stego pairs (8000 for train and 2000 for validation) can achieve a good effect. Thus, for a quick experiments, you do not need to many audio samples in the design of the network.

## Steganographic Algorithm
**HCM**<sup>[1,2]</sup> (Huffman Code Mapping) and **EECS**<sup>[3]</sup> (Equal Length Entropy Codes Substitution, an adaptive MP3 steganographic algorithm with [**STC**](http://dde.binghamton.edu/download/syndrome/) and **distortion function** based on **psychological acoustics model** (**PAM**)).

Note: All **built-in** MP3 algorithms embeds secret messages in the process of MP3 encoding, which will change QMDCT coefficients of MP3. So, this network can be applied to detect all data hiding methods which have impact on the QMDCT coefficients.

All steganographic algorithms are coded by **Kun Yang** (E-Mail: yangkun9076@iie.ac.cn)

## Steganalytic Algorithms based on Traditional Feature Sets
All steganalysis algorithms are available in our another repository -- [audio_steganalysis_ml](https://github.com/Charleswyt/audio_steganalysis_ml).
* D2MA<sup>[4]</sup>
* MDI2<sup>[5]</sup>
* ADOTP<sup>[6]</sup>

## Steganalytic Algorithms based on Deep Learning
* [WASDN](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/CNN-based%20Steganalysis%20of%20MP3%20Steganography%20in%20the%20Entropy%20Code%20Domain)
* [RHFCN](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/RHFCN%EF%BC%9AFully%20CNN-based%20Steganalysis%20of%20MP3%20with%20Rich%20High-Pass%20Filtering)

## Reference
**[1]** Haiying Gao. 2007. [**The MP3 Steganography Algorithm based on Huffman Coding**](https://www.researchgate.net/publication/290779951_The_MP3_steganography_algorithm_based_on_huffman_coding). Acta Scientiarum Naturalium Universitatis Sunyatseni 4 (2007), 009. <br>
**[2]** Diqun Yan, Rangding Wang, and Liguang Zhang. 2011. [**A High Capacity MP3 Steganography based on Huffman Coding**](http://xueshu.baidu.com/s?wd=paperuri%3A%2847ca19607f5dfdde6cbc1fca4f6dc5ad%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fen.cnki.com.cn%2FArticle_en%2FCJFDTotal-SCDX201106013.htm&ie=utf-8&sc_us=17794155201621866322). Journal of Sichuan University (Natural Science Edition) 6 (2011), 013. <br>
**[3]** Kun Yang, Xiaowei Yi, Xianfeng Zhao, and Linna Zhou. 2017. [**Adaptive MP3 Steganography Using Equal Length Entropy Codes Substitution**](https://link.springer.com/chapter/10.1007/978-3-319-64185-0_16). In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 202–216. <br>
**[4]** Mengyu Qiao, Andrew H. Sung, and Qingzhong Liu. 2013. [**MP3 Audio Steganalysis**](http://xueshu.baidu.com/s?wd=paperuri%3A%28baa2297b4d905e182d8c02ea52851247%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fdl.acm.org%2Fcitation.cfm%3Fid%3D2442161.2442240&ie=utf-8&sc_us=14226838812282894210). Information Sciences, 231:123-134, 2013. <br>
**[5]** Yanzhen Ren, Qiaochu Xiong, and Lina Wang. 2017. [**A Steganalysis Scheme for AAC Audio Based on MDCT Difference Between Intra and Inter Frame**](https://link.springer.com/chapter/10.1007%2F978-3-319-64185-0_17). In Digital Forensics and Watermarking - 16th International Workshop, IWDW 2017, Magdeburg, Germany, August 23-25, 2017, Proceedings. 217–231. <br>
**[6]** Chao Jin, Rangding Wang, and Diqun Yan. 2017. [**Steganalysis of MP3Stego with Low Embedding-Rate using Markov Feature**](https://link.springer.com/article/10.1007%2Fs11042-016-3264-y). Multimedia Tools and Applications 76, 5 (2017), 6143–6158. <br>
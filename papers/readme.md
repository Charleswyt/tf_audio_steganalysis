## Dataset
The dataset can be downloaded from the [link](https://pan.baidu.com/s/1rYCzJRksHkgbOOYI9MqQjA), and we name it as **Audio Steganalysis Dataset, IIE (ASDIIE)**.

The ***extraction password*** is "**z28d**".

A pure dataset which only contains WAV audio files and audio encoders is available through [**IEEEDataPort**](http://ieee-dataport.org/documents/audio-steganalysis-dataset).

We divide our dataset into **vocal songs** and **light music,** and we mainly steganalyze **vocal songs** which are with more complex waveform.

---

Besides, 16,000 [**BBC Sound Effects**](http://bbcsfx.acropolis.org.uk/) are made available by the **BBC** in **WAV** format to download for use under the terms of the [RemArc Licence](https://github.com/bbcarchdev/Remarc/raw/master/doc/2016.09.27_RemArc_Content%20licence_Terms%20of%20Use_final.pdf).

There are some other [**audio databases**](https://github.com/Charleswyt/tf_audio_steganalysis/paper/audio_database.md) that we can utilize.

For a **quick** experiments, you do not need to many audio samples in the design of the network. According to our experiments, **10000** cover-stego audio pairs (8000 for training and 2000 for validation) can achieve good performance.

## Steganographic Algorithm
**HCM**<sup>[1,2]</sup> (Huffman Code Mapping) and **EECS**<sup>[3]</sup> (Equal Length Entropy Codes Substitution, an adaptive MP3 steganographic algorithm with [**Syndrome-Trellis Code (STC)**](http://dde.binghamton.edu/download/syndrome/)<sup>[4]</sup> and **distortion function** based on **psychological acoustics model** (**PAM**)).

Note: All **built-in** MP3 algorithms embeds secret messages in the process of MP3 encoding, which will change QMDCT coefficients of MP3. So, this network can be applied to detect all data hiding methods which have impact on the QMDCT coefficients.

All steganographic algorithms are coded by **Kun Yang** (E-Mail: yangkun9076@iie.ac.cn)

## Steganalytic Algorithms based on Traditional Handcrafted Feature Design
All steganalysis algorithms are available in our another repository -- [audio_steganalysis_ml](https://github.com/Charleswyt/audio_steganalysis_ml).
* D2MA<sup>[5]</sup>
* MDI2<sup>[6]</sup>
* ADOTP<sup>[7]</sup>
---
* I2C<sup>[8]</sup>
* Co-Occurrence Matrix<sup>[9]</sup>

## Steganalytic Algorithms based on Deep Neural Network Design
* [WASDN](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/CNN-based%20Steganalysis%20of%20MP3%20Steganography%20in%20the%20Entropy%20Code%20Domain)<sup>[10]</sup>
* [RHFCN](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/papers/RHFCN%20-%20Fully%20CNN-based%20Steganalysis%20of%20MP3%20with%20Rich%20High-Pass%20Filtering)<sup>[11]</sup>

## Reference
**[1]** GAO H. [**The MP3 Steganography Algorithm based on Huffman Coding**](https://www.researchgate.net/publication/290779951_The_MP3_steganography_algorithm_based_on_huffman_coding)[J]. Acta Scientiarum Naturalium Universitatis Sunyatseni, 2007, 46(4): 32-35. <br>
**[2]** Yan D, WANG R, ZHANG L. 2011. [**A High Capacity MP3 Steganography based on Huffman Coding**](http://xueshu.baidu.com/s?wd=paperuri%3A%2847ca19607f5dfdde6cbc1fca4f6dc5ad%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fen.cnki.com.cn%2FArticle_en%2FCJFDTotal-SCDX201106013.htm&ie=utf-8&sc_us=17794155201621866322)[J]. Journal of Sichuan University (Natural Science Edition), 2011, 48(6): 1281-1286. <br>
**[3]** YANG K, YI X, ZHAO X, ZHOU L. 2017. [**Adaptive MP3 Steganography Using Equal Length Entropy Codes Substitution**](https://link.springer.com/chapter/10.1007/978-3-319-64185-0_16)[C]. Proceedings of the 16th International Workshop on Digital Forensics and Watermarking (IWDW 2017). Magdeburg, Germany: Springer, August 23-25, 2017: 202–216. <br>
**[4]** FILLER T, JUDAS J, FRIDRICH J. [**Minimizing Additive Distortion in Steganography using Syndrome-Trellis Codes**](https://ieeexplore.ieee.org/document/5740590)[J]. IEEE Transactions on Information Forensics and Security (TIFS), 2011, 6(3): 920--935. <br>
**[5]** QIAO M, SUNG A H, LIU Q. 2013. [**MP3 Audio Steganalysis**](http://xueshu.baidu.com/s?wd=paperuri%3A%28baa2297b4d905e182d8c02ea52851247%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fdl.acm.org%2Fcitation.cfm%3Fid%3D2442161.2442240&ie=utf-8&sc_us=14226838812282894210)[J]. Information Sciences, 2013, 231:123-134. <br>
**[6]** REN Y, XIONG Q, WANG L. 2017. [**A Steganalysis Scheme for AAC Audio Based on MDCT Difference Between Intra and Inter Frame**](https://link.springer.com/chapter/10.1007%2F978-3-319-64185-0_17)[C]. Proceedings of the 16th International Workshop on Digital Forensics and Watermarking (IWDW 2017). Magdeburg, Germany: Springer, August 23-25, 2017: 217–231. <br>
**[7]** JIN C, WANG R, YAN D. 2017. [**Steganalysis of MP3Stego with Low Embedding-Rate using Markov Feature**](https://link.springer.com/article/10.1007%2Fs11042-016-3264-y)[J]. Multimedia Tools and Applications (MTAP), 2017, 76(5): 6143–6158. <br>
**[8]** WANG Y, YANG K, YI X, ZHAO X. [**MP3 Audio Steganalysis Utilizing both Intrablock and Interblock Correlations**](http://www.media-security.net/?p=976)[C]. Proceedings of 14th China Information Hiding Workshop (CIHW 2018). Guangzhou, China, March 31 - April 1, 2018: 829-836. <br>
**[9]** 王让定, 羊开云, 严迪群, 金超, 孙冉, 周劲蕾. [**一种基于共生矩阵分析的MP3音频隐写检测方法**](http://cprs.patentstar.com.cn/Search/Detail?ANE=9DEA9CIB7CEA7ACA9BHA9EID9GEB9IDH9EECACGADFIA4DBA)[P]. 2015.05.20.
**[10]** WANG Y, YANG K, YI X, ZHAO X, XU Z. [**CNN-based Steganalysis of MP3 Steganography in the Entropy Code Domain**](https://dl.acm.org/citation.cfm?id=3206011)[C]. Proceedings of the 6th ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec 2018). Innsbruck, Austria: ACM, June 20-22, 2018: 55-65. <br>
**[11]** WANG Y, YI X, ZHAO X, SU A. [**RHFCN: Fully CNN-based Steganalysis of MP3 with Rich High-pass Filtering**](https://ieeexplore.ieee.org/document/8683626)[C]. Proceedings of the 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2019). Brighton, UK: IEEE, May 12-17, 2019: 2627-2631. <br>

P.S. In order to facilitate your reading of the above papers, you can download all these papers through the [link](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/papers).

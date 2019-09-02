# Audio Steganalysis with Deep Learning

@ Author: **[Yuntao Wang (Charles_wyt)](http://www.escience.cn/people/wangyuntao/index.html)** <br>
@ Email: wangyuntao2@iie.ac.cn <br>
Hope we have a happy communication.

| **`Linux CPU`** | **`Linux GPU`** | **`Windows CPU`** | **`Windows GPU`** |
|-|-|-|-|
|![Travis](https://img.shields.io/travis/rust-lang/rust/master.svg)  |![Travis](https://img.shields.io/travis/rust-lang/rust/master.svg)    |![Travis](https://img.shields.io/travis/rust-lang/rust/master.svg)    |![Travis](https://img.shields.io/travis/rust-lang/rust/master.svg)    |
---
This project is a **tensorflow** implementation of our recent work, and you can design your own networks through the platform.
+ [CNN-based Steganalysis of MP3 Steganography in the Entropy
Code Domain](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/papers/CNN-based%20Steganalysis%20of%20MP3%20Steganography%20in%20the%20Entropy%20Code%20Domain) [[IH&MMSec](https://www.ihmmsec.org) 2018, Best Paper Award] <br>
**[[Paper (ACM)](https://dl.acm.org/citation.cfm?id=3206011)]** **[[Paper (pdf)](http://www.media-security.net/?p=809)]** **[[Dataset](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/papers)]**
+ [RHFCN: Fully CNN-based Steganalysis of MP3 with Rich High-Pass Filtering](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/papers/RHFCN%20-%20Fully%20CNN-based%20Steganalysis%20of%20MP3%20with%20Rich%20High-Pass%20Filtering) [[ICASSP](https://2019.ieeeicassp.org) 2019] <br>
**[[Paper (IEEE)](https://ieeexplore.ieee.org/document/8683626)]** **[[Paper (pdf)](http://www.media-security.net/?p=969)]** **[[Dataset](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/papers)]**

## Necessary Packages
tensorflow-gpu==1.3 or later, numpy, pandas, matplotlib, scikit-image, scikit-learn, filetype, [virtualenv](https://charleswyt.github.io/2018/09/06/python%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%E5%8F%8A%E4%BD%BF%E7%94%A8/), [librosa](http://librosa.github.io/librosa/core.html) (depends on **[FFmpeg](http://www.ffmpeg.org/download.html)**)

You can use the command, **pip install -r requirements.txt**, to install all necessary packages mentioned above. If you don't want to change or break your original version of tensorflow, you can use [**virtualenv**](https://charleswyt.github.io/2018/09/06/python%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%E5%8F%8A%E4%BD%BF%E7%94%A8/) to create a new python runtime environment.

## How to Use
1. Install [**Python3.x**](https://www.python.org/) or [**Anaconda**](https://repo.continuum.io/archive/), and add the installation directory into the environment variable (recommand python3.5).
2. **GPU** runtime environment [**configure**](https://blog.csdn.net/yhaolpz/article/details/71375762?locationNum=14&fps=1) for the network training (**optional**).
3. Install all dependent packages mentioned above (open **[setup](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/setup)/requirements.txt**, and input "**pip install -r requirements**" into your terminal window).
4. **Run** the code as the [**example**](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/src/config_file) shows.
5. Use [**tensorboard**](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/graph_viz.html) to visualize the training process such as the **accuracy** and **loss curve** of the training. The command is "**tensorboard --logdir=/path of log**".
6. If you want to design your own network based on this project, there is a brief [**instruction**](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/src/networks/network_design.md) for you.
7. All our sourcecode is writen with [**Pycharm**](https://github.com/Charleswyt/tf_audio_steganalysis/blob/master/setup/pycharm.md), and the **hard wrap** is setted as **180**. If your setting of hard wrap is less than 180, there will be warnings shwon in the IDE.

## Files Description
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   audio_samples           |   some audio samples
2       |   data_processing         |   dataset build, tools which are used for QMDCT coefficients extraction and so on
3       |   jupyter                 |   a folder for debug with jupyter
4       |   papers                  |   the paper, presentation, dataset and brief introduction of our recent work
5       |   setup                   |   setup and configuration
6       |   src                     |   source code

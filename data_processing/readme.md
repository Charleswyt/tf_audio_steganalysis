## Tools for Audio Steganalysis

### Files Description
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   batch_scripts           |   batch scripts for the run of python scripts
2       |   python_scripts          |   python scripts for cover and stego audio samples make, QMDCT coefficients matrix extraction, and so on
3       |   secret_message          |   secret message files for embedding
4       |   BatchRename.exe         |   an executable program for files rename in batch

### Procedure of Handcrafted Feature Design
1. Audio samples make.
2. Audio samples check (bitrate, file size, steganographic algorithms, embedding rate).
3. QMDCT coefficients matrix extraction (saved in *.txt format).
4. Train(, validation) and test datasets split (Optional).
5. [Feature design and extraction](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/feature_extract).
6. Steganalysis with traditional classifier ([SVM or ensemmble classifier](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/train_test)).
7. Record and analyze the experimental results.

### Procedure of Deep Network Design
1. Audio samples make.
2. Audio samples check (bitrate, file size, steganographic algorithms, embedding rate).
3. QMDCT coefficients matrix extraction (saved in *.txt format).
4. Train(, validation) and test datasets split (Optional).
5. [Steganalysis with deep learning-based method](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper).
6. Record and analyze the experimental results.


![](https://i.imgur.com/kxJg8HC.jpg)
## Tools for steganalysis

### File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   batch_scripts           |   batch scripts for run of python scripts
2       |   python_scripts          |   python scripts for samples make, QMDCT extraction and so on
3       |   secret_message          |   secret message files for embedding
4       |   BatchRename.exe         |   a *.exe for file rename in batch

### Procedure of data processing (machine learning)
1. samples make
2. samples check (bitrate, file size, steganographic algorithms, embedding rate)
3. QMDCT extraction
4. train and test sets split
5. [feature extraction](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/feature_extract)
6. steganalysis through traditional classifier ([SVM or ensemmble classifier](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/train_test))
7. record the results

### Procedure of data processing (deep learning)
1. samples make
2. samples check (bitrate, file size, steganographic algorithms, embedding rate)
3. QMDCT extraction
4. train and test sets split
5. [steganalysis through deep learning-based method](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper)
6. record the results

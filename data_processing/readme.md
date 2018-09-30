## Tools for steganalysis

### File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   batch_scripts           |   batch scripts for run of python scripts
2       |   python_scripts          |   python scripts for samples make, QMDCT extraction and so on
3       |   secret_message          |   secret message file for embedding
4       |   BatchRename.exe         |   an *.exe for file rename in batch

### Procedure of data processing (machine learning)
1. samples making
2. samples check (bitrate, file size, embedding rate)
3. QMDCT extraction
4. train and test split
5. [feature extraction](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/feature_extract)
6. steganalysis via traditional classifier ([SVM or ensemmble classifier](https://github.com/Charleswyt/audio_steganalysis_ml/tree/master/train_test))
7. record the results

### Procedure of data processing (deep learning)
1. samples making
2. samples check (bitrate, file size, embedding rate)
3. QMDCT extraction
4. train and test split
5. [steganalysis via deep learning-based method](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/paper/CNN-based%20Steganalysis%20of%20MP3%20Steganography%20in%20the%20Entropy%20Code%20Domain)
6. record the results

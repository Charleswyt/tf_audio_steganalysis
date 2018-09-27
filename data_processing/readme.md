## Tools for steganalysis

### File description
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   batch_scripts           |   batch scripts for run of python scripts
2       |   python_scripts          |   python scripts for samples make, QMDCT extraction and so on
3       |   secret_message          |   secret message file for embedding
4       |   BatchRename.exe         |   an *.exe for file rename in batch

### Procedure of data processing
1. samples make
2. samples check (bitrate, file size, embedding rate)
3. QMDCT extraction
4. train and test split
5. feature extraction
6. SVM train, validation and test
7. record the results
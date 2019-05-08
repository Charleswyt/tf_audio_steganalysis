### List of Python Scripts
ID      |   File                    |   Function
:-      |   :-                      |    :-
1       |   batch_scripts           |   batch scripts for run of python scripts
2       |   python_scripts          |   python scripts for samples make, QMDCT extraction and so on
3       |   secret_message          |   secret message file for embedding
4       |   BatchRename.exe         |   an *.exe for file rename in batch


## Usage of Python Scripts

### QMDCT extraction
python QMDCT_extract.py "files_path" "file_num"
+ **files_path**: path of QMDCT coefficients matrix files
+ **file_num**: num of files to be extracted, default: None (all files in this path are extracted)

usage: 
1. python QMDCT_extract.py ***
2. python QMDCT_extract.py *** 1000

### File move (Optional)
The extracted QMDCT coefficients files and audio files are kept in the same path. To move these coefficients files, you need this python scripts.

python file_move.py "root_old" "root_new" "file_type"
+ **files_path**: original file path
+ **file_num**: new file path
+ **file_type**: type of files to be moved, default: txt
  
usage: 
1. python file_move.py *** ***
2. python file_move.py *** *** txt

### Train and test dataset split (Optional)
If you need to split the dataset into train and test, this python script may satisfy you.

python file_move.py "files_path" "percent_train" "percent_validation"
+ **files_path**: file path of dataset
+ **percent_train**: percent of train dataset (default: 0.7)
+ **percent_validation**: percent of valiadation dataset (default: 0.3)

usage:
1. python train_test_split.py ***
2. python train_test_split.py *** 0.5
3. python train_test_split.py *** 0.6 0.4

If the sum of two percent is not 1.0, an error is presented.

### Tools for Samples Make
The usage of each encoder is shown as follows.

Encoder         |   Command
:-:      	    |    :-
Encode          |   encode.exe -b "bitrate" "path of wav audio" "path of mp3 audio"
MP3Stego(cover) |   encode_MP3Stego.exe -b "bitrate"
MP3Stego(stego) |   encode_MP3Stego.exe -b "bitrate" -E "path of embedding file" -P "password" "path of wav audio" "path of mp3 audio"
HCM             |   encode_HCM.exe -b "bitrate" -embed "path of embedding file" -cost "type of cost function" -er "embedding_rate" -framenumber "maximum number of embedding frames" "path of wav audio" "path of mp3 audio"

There are points which are needed to be paid attention.
+ MP3Stego can be used to encode MP3, however the structure of LAME and MP3Stego is not fully the same. Thus, the cover samples of HCM, EECS and others are encoded via LAME and the cover samples of MP3Stego are encoder via MP3Stego.
+ The MP3Stego encoder is not reformed by us, which means we can't decide the length of secret messages via command line. And we complete the change of message length via the change of embedding file path. 
+ The meaning of each parameter are:
    * -**b**: bitrate, 128, 192, 256, 320.
    * -**E**: path of embedding file, any valid path is okay, just for MP3Stego.
    * -**P**: password, just for MP3Stego.
    * -**embed**: path of embedding file, any valid path is okay.
    * -**er**: embedding rate, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
    * -**framenum**: maximum number of embedding frames, 50 in general.
    * -**cost**: type of cost function, 1, 2
    * -**width**: width of parity-check matrix, 2, 3, 4, 5, 6, 7, 8, 9, ..., just for algorithm with STC
    * -**height**: height of parity-check matrix, 7 in general, just for algorithm with STC.
+ If you use the encoder shown above, replace the content in "\*\*\*" is OK.

## Usage of Python Scripts

### QMDCT Coefficients Matrix Extraction
#### Function
QMDCT coefficients matrix is an important data format for MP3 steganalysis. This script is used to extract coefficients matrix of MP3 audio files in batch.

#### Usage
> python QMDCT\_extraction.py "files\_path" "file\_num"

#### Parameters
+ **files_path**: the path of QMDCT coefficients matrix files
+ **file_num**: the number of files to be extracted, default: None (all files in this path are extracted)

#### Examples
1. python QMDCT_extract.py ***
2. python QMDCT_extract.py *** 1000

### File Move (Optional)
#### Function
The extracted QMDCT coefficients matrix files and audio files are saved in the same directory. To separate these coefficients files from the original folder, you may need this python scripts.

> python file\_move.py "directory\_old" "directory\_new" "file\_type"

#### Parameters
+ **files_path**: the original file path
+ **file_num**: the new file path
+ **file_type**: the type of files to be moved, default: txt
  
#### Examples
1. python file_move.py *** ***
2. python file_move.py *** *** txt

### Train and Test Datasets Split (Optional)
#### Function
If you need to split the dataset into train and test, this python script may satisfy you. If you use "json-semi" mode, you need not split the dataset.

#### Usage
> python file\_move.py "files\_path" "percent\_train" "percent\_validation"

#### Parameters
+ **files_path**: the file path of the whole dataset
+ **percent_train**: the percent of train dataset (default: 0.7)
+ **percent_validation**: the percent of valiadation dataset (default: 0.3)

#### Usage
1. python train_test_split.py ***
2. python train_test_split.py *** 0.5
3. python train_test_split.py *** 0.6 0.4

If the sum of two percents is not 1.0, an error is presented.

### Tools for Audio Samples Make
The usage of each encoder is shown as follows.

Encoder         |   Usage
:-:      	    |   :-
Encode          |   encode.exe -**b** "bitrate" "path of wav audio" "path of mp3 audio"
MP3Stego(cover) |   encode_MP3Stego.exe -**b** "bitrate"
MP3Stego(stego) |   encode_MP3Stego.exe -**b** "bitrate" -**E** "path of embedding file" -**P** "password" "path of wav audio" "path of mp3 audio"
HCM             |   encode_HCM.exe -**b** "bitrate" -**embed** "path of embedding file" -**cost** "type of cost function" -**er** "embedding_rate" -**framenumber** "maximum number of embedding frames" "path of wav audio" "path of mp3 audio"
EECS            |   encode_EECS.exe -**b** "bitrate" -**embed** "path of embedding file" -**width** "the width of parity-check matrix" -**height** "the height of parity-check matrix" -**key** "scrambling key" -**er** "embedding_rate" -**framenumber** "maximum number of embedding frames" "path of wav audio" "path of mp3 audio"

There are points which are needed to be paid attention.
+ MP3Stego encoder is developed based on the 8Hz MP3 encoder, which can be used to encode MP3, however, the structure of LAME and MP3Stego is not fully the same. Thus, the cover samples of HCM, EECS and others are encoded via LAME and the cover samples of MP3Stego are encoder via MP3Stego.
+ The MP3Stego encoder is not reformed by us, which means we can't decide the length of secret messages via command line. And we complete the change of message length via the change of embedding file path. 
+ The meaning of each parameter are listed as:
    * -**b**: bitrate, 128, 192, 256, 320, etc.
    * -**E**: path of embedding files, any valid path is okay, just used in MP3Stego.
    * -**P**: password, just used in MP3Stego.
    * -**embed**: path of embedding file, any validation path is okay.
    * -**er**: relative embedding rate (steganographic frames rate), **0.1 : 0.1 : 1.0.**
    * -**framenumber**: maximum number of embedding frames, 50 in general.
    * -**cost**: the type of cost function, 1, 2
    * -**width**: the width of parity-check matrix, 2, 3, 4, 5, 6, 7, 8, 9, ..., just used in the algorithm with STC.
    * -**height**: the height of parity-check matrix, 7 in general, just used in the algorithm with STC.
    * -**key**: scrambling key.
+ If you use the encoder shown above, replace the content in "\*\*\*" is OK.

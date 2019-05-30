### Files Description
The usage of each encoder is shown as follows.

Encoder             |   Description
:-:      	        |   :-
encode.exe          |   lame encoder
encode_MP3Stego.exe |   MP3Stego encoder for MP3Stego cover and stego
encode_HCM.exe      |   HCM encoder for MP3Stego cover and stego
lame_qmdct.exe      |   QMDCT coefficients matrix extractor

### Tools for Audio Samples Make
The usage of each encoder is shown as follows.

Encoder         |   Usage
:-:      	    |   :-
Encode          |   encode.exe -**b** "bitrate" "path of wav audio" "path of mp3 audio"
MP3Stego(cover) |   encode_MP3Stego.exe -**b** "bitrate"
MP3Stego(stego) |   encode_MP3Stego.exe -**b** "bitrate" -**E** "path of embedding file" -**P** "password" "path of wav audio" "path of mp3 audio"
HCM             |   encode_HCM.exe -**b** "bitrate" -**embed** "path of embedding file" -**cost** "type of cost function" -**er** "embedding_rate" -**framenumber** "maximum number of embedding frames" "path of wav audio" "path of mp3 audio"
EECS            |   encode_EECS.exe -**b** "bitrate" -**embed** "path of embedding file" -**width** "the width of parity-check matrix" -**height** "the height of parity-check matrix" -**key** "scrambling key" -**er** "embedding_rate" -**framenumber** "maximum number of embedding frames" "path of wav audio" "path of mp3 audio"

The meaning of each parameter are listed as:
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
## How to design your own networks through the template

1. Create a new python script named **my_own_networks** in the folder of ../src/**networks**.
2. Import necessary package.
   
        from layer import *
        from kernels import *
    
3. Design your own networks.
4. Import your own network scipt in "../src/networks/**networks.py**", and add your name of designed network into networks list.
5. Modify the json file for training (../src/config_file/**config_train.json**). There is a brief [introduction](https://github.com/Charleswyt/tf_audio_steganalysis/tree/master/src/config_file) for the usage of config files.
6. Run the code.
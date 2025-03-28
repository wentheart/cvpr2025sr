# cvpr2025sr
## new
The python version is 12.9 and the CUDA version is 12.6.<br><br>
Put the LR data of the test set in ./data/DIV2K_test/LR folder.<br><br>
You can easily generate SR images by running the following commands:
```
python test.py --test_dir ./data/DIV2K_test_LR --model_id 27
```
The path of all output images uses the default path under the official framework.<br><br>


## old(but also recmmonded to read)
This github repository is used to submit 4x Super Resolution contests.

inference.py files are generated with super-resolution images using the trained model.<br>
<br>
The parameter file of the trained model is stored in the checkpoints folder.<br>
<br>
The path to the resulting super-resolution image is: ./baseModelFusion/results/test/...you can see the details in inference.py.<br>
<br>
baseModelFusionNet.py is the main file of the fusion model, and the model used for fusion is located in the models folder.<br>
<br>
model_zoo folder stores the parameter files of the fine-tuned baseline model.<br>
<br>
The data folder is used to store datasets.
<br>
<br>
The test output image that has been submitted to Codalab can be downloaded from the following link:
<br>
```bash

wget https://github.com//wentheart/sr_output/releases/download/v1.0/baseFusionOutput.zip
```
The factsheet folder stores the relevant factsheet files.
<br>
<br>
The Real_ESRGAN and trans_ganModelFusion folders were miscellaneous folders that were used to verify the effectiveness of another fusion scheme, but were not ultimately adopted.<br>
<br>

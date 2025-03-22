# cvpr2025sr
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
The test output image that has been submitted to Codalab can be downloaded from the following link:
'''
<br>
```bash

wget https://github.com//wentheart/sr_output/releases/download/v1.0/baseFusionOutput.zip
```
<br>
'''
<br>
The Real_ESRGAN and trans_ganModelFusion folders were miscellaneous folders that were used to verify the effectiveness of another fusion scheme, but were not ultimately adopted.<br>
<br>

# DRML
## File Structure 
  Project/  
    &nbsp;&nbsp;&nbsp;&nbsp;env_file/  
    &nbsp;&nbsp;&nbsp;&nbsp;DRML/  

## Create env
  `python -m venv env_file`

If it gives some error, execute:
   `Set-ExecutionPolicy Unrestricted -Scope Process`

Activate env:
   `.\env_file\Scripts\activate.Ps1`


## Installation  of packages
  `pip install -r requirements.txt`
  
To install gpu version: 
  `pip install -r .\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`


## Preprocessing dataset
  In the file tool/ provide files to processing image and label of dataset DISFA+ and CK+.
  
## Experiments
###Parameters
optimizer：SGD
learning rate：0.001
weight decay：0.005
momentum：0.9
epoch：20
batch size：64

### Results in DISFA+ 
Using evaluation metrics: Accuracy/F1-Score 

| AU1 | AU2 | AU4 |	AU6	| AU9 |	AU12| AU25 | AU26 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|68.96/90.98|	70.94/93.51|	74.79/94.61|	64.90/91.56|	66.20/97.02	|79.50/95.33|91.61/96.45|	74.33/94.38	|
 

| AU1 | AU2 | AU4 |	AU5 |	AU6	| AU9 |	AU12| AU15 | AU17 | AU20 | AU25 | AU26 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|68.92/90.36|	72.59/92.72|	60.95/92.70|	64.91/84.80|	69.52/92.71	|62.61/96.70|	85.82/96.79	|43.78/95.61|	25.99/94.09	|22.48/96.91	|94.92/97.88|76.50/94.14	|62.42/93.79|

### Reference
[ Deep Region and Multi-label Learning for Facial Action Unit Detection](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf)  
[ Official DMRL code](https://github.com/zkl20061823/DRML)  
[ Pytorch DMRL code](https://github.com/AlexHex7/DRML_pytorch)  

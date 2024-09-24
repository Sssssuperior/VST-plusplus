# VST++
This is an official implementation for “VST++: Efficient and Stronger Visual Saliency Transformer”

Nian Liu, Ziyang Luo, Ni Zhang, Junwei Han

⁉️ **Here is just an initial version, and I haven't spent much time organizing the code since it's quite old. If you find any issues, please let me know, and I'll make the necessary adjustments promptly.**

## Environmental Setups
Pytorch $\geq$ 1.6.0, Torchvision $\geq$ 0.7.0

## Data Preparation
### RGB SOD & RGB-D SOD
For RGB SOD and RGB-D SOD, we employ the following datasets to train our model concurrently: the training set of **DUTS** for `RGB SOD` , the training sets of **NJUD**, **NLPR**, and **DUTLF-Depth** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**, and **SOD**, while **STERE**, **NJUD**, **NLPR**, **DUTLF-Depth**, **SIP**, **LFSD**, **RGBD135**, **SSD** and **ReDWeb-S** datasets are employed for testing the RGB-D SOD task. You can directly download these datasets by following [[VST]](https://github.com/nnizhang/VST?tab=readme-ov-file).

## Experiments
First, you should choose the specific backbone version in Models/ImageDepthNet.py. For swin version, we provide swin_transformer_T, swin_transformer_S, swin_transformer_B, and for T2T-ViT, the T2t_vit_t_14 version is provided.

Then, some paths in  train_test_eval.py shold be change. For example, '--data_root' and '--pretrained_model'. 

Finally, run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation which is similar to VST.

Please note that we provide two versions of the code here. The first version is the original version, which can be trained and tested, and the second version is the version of select token. 

Please refer to the Select-Integrate Attention section of the paper for details.

`It is essential to acknowledge that the training method of our SIA differs from that of testing. Due to the uncertainty  regarding the number of selected foreground patches, which hinders parallel computation, we still use all patch tokens and adopt the masked attention to filter out background patch tokens during training.
`

## Results

### 1. Model Zoo
| Name | Backbone | RGB Weight | RGB-D Weight | 
|  :---: |  :---:    | :---:   |  :---:   |
| VST++-t |  T2T-ViT    |  [[baidu](https://pan.baidu.com/s/1h4tV4i6fL8pvwkMrKQcWCA?pwd=sbra),PIN:sbra]/ [[Geogle Drive](https://drive.google.com/file/d/14CIcPEH4w9gcL3C-NWzdF26n7eCYDMJF/view?usp=sharing)]   | [[baidu](https://pan.baidu.com/s/17BrdgGtCrDZnO76qIcFIrg?pwd=hzr1),PIN:hzr1]/ [[Geogle Drive](https://drive.google.com/file/d/1Z9AWqyjJOmo2trXCOXOZKGsHgUcKZdr_/view?usp=sharing)] |
| VST++-T |  Swin-T    | [[baidu](https://pan.baidu.com/s/1dGf6fAjHmP3tCwJIuHv1AA?pwd=cbfm),PIN:cbfm]/ [[Geogle Drive](https://drive.google.com/file/d/1CtovuG63Xal7H-RaMh7arpg4mUkl5YAP/view?usp=sharing)]   | [[baidu](https://pan.baidu.com/s/188LO0l5ki70tkW5r0l5sgw?pwd=69dv),PIN:69dv]/ [[Geogle Drive](https://drive.google.com/file/d/1C3Erb8aq0-f_75so6mgVR6-0yNDC-CPD/view?usp=sharing)] |
| VST++-S |  Swin-S    | [[baidu](https://pan.baidu.com/s/1zzh9tewf6fYxZ6lNe7j0YA?pwd=g5oc),PIN:g5oc]/ [[Geogle Drive](https://drive.google.com/file/d/1wYLzQZy-xPddKde6tUSEE1BTrxrgoX_w/view?usp=sharing)]    | [[baidu](https://pan.baidu.com/s/1pUikzhuvLLpwAf1aVIqpwA?pwd=8vcz),PIN:8vcz]/ [[Geogle Drive](https://drive.google.com/file/d/1SbQU3nqPPCqxNYvq-YfUUEIEiiX7nRO2/view?usp=sharing)] |
| VST++-B |  Swin-B    |  [[baidu](https://pan.baidu.com/s/1cpuOj6jIwdld57UkDGMSfg?pwd=5t0v),PIN:5t0v]/ [[Geogle Drive](https://drive.google.com/file/d/1JhOFgsxlSWhhgOvvGzhHl2HcpJhuD_TG/view?usp=sharing)]  | [[baidu](https://pan.baidu.com/s/139QkBzeWU_FT1DJR8om1wA?pwd=ug9y),PIN:ug9y]/ [[Geogle Drive](https://drive.google.com/file/d/1QvopABA4NR5MuWSiXBCQH__zBZbPDTmL/view?usp=sharing)]|

### 2. Prediction Maps
We offer the prediction maps of **RGB** task [[baidu](https://pan.baidu.com/s/1KWLpliMzXDyOP7OmtmFVoA?pwd=e5sl),PIN:e5sl]/ [[Geogle Drive](https://drive.google.com/drive/folders/1ll3cALBffbt9FOKBtY-7UOxcYfFcsMUM?usp=sharing)] and **RGB-D** task [[baidu](https://pan.baidu.com/s/1GcT3gRW_sE-fIWjcMn-91g?pwd=ns1e),PIN:ns1e]/ [[Geogle Drive](https://drive.google.com/drive/folders/1E7eWEODUs1_Rl0Rc5PSj-pt-n6GHzWlT?usp=sharing)] at this time for all backbones.

## Citation
If you use VST++ in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
```
@article{liu2024vst++,
  title={Vst++: Efficient and stronger visual saliency transformer},
  author={Liu, Nian and Luo, Ziyang and Zhang, Ni and Han, Junwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

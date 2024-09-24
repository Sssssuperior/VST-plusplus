# VST++
This is an official implementation for “VST++: Efficient and Stronger Visual Saliency Transformer”
Nian Liu, Ziyang Luo, Ni Zhang, Junwei Han


## Environmental Setups
Pytorch $\geq$ 1.6.0, Torchvision $\geq$ 0.7.0

## Data Preparation
### RGB SOD & RGB-D SOD
For RGB SOD and RGB-D SOD, we employ the following datasets to train our model concurrently: the training set of **DUTS** for `RGB SOD` , the training sets of **NJUD**, **NLPR**, and **DUTLF-Depth** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**, and **SOD**, while **STERE**, **NJUD**, **NLPR**, **DUTLF-Depth**, **SIP**, **LFSD**, **RGBD135**, **SSD** and **ReDWeb-S** datasets are employed for testing the RGB-D SOD task. You can directly download these datasets by following [[VST]](https://github.com/nnizhang/VST?tab=readme-ov-file).

## Experiments
Run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation which is similar to VST.

## Results

### 1. Model Zoo
| Name | Backbone | RGB Weight | RGB-D Weight | 
|  :---: |  :---:    | :---:   |  :---:   |
| VST++-t |  T2T-ViT    |  [[baidu](https://pan.baidu.com/s/1h4tV4i6fL8pvwkMrKQcWCA?pwd=sbra),PIN:sbra]/ [[Geogle Drive](https://drive.google.com/file/d/14CIcPEH4w9gcL3C-NWzdF26n7eCYDMJF/view?usp=sharing)]   | [[baidu](),PIN:]/ [[Geogle Drive]()] |
| VST++-T |  Swin-T    | [[baidu](https://pan.baidu.com/s/1dGf6fAjHmP3tCwJIuHv1AA?pwd=cbfm),PIN:cbfm]/ [[Geogle Drive](https://drive.google.com/file/d/1CtovuG63Xal7H-RaMh7arpg4mUkl5YAP/view?usp=sharing)]   | [[baidu](),PIN:]/ [[Geogle Drive]()] |
| VST++-S |  Swin-S    | [[baidu](https://pan.baidu.com/s/1zzh9tewf6fYxZ6lNe7j0YA?pwd=g5oc),PIN:g5oc]/ [[Geogle Drive](https://drive.google.com/file/d/1wYLzQZy-xPddKde6tUSEE1BTrxrgoX_w/view?usp=sharing)]    | [[baidu](),PIN:]/ [[Geogle Drive]()] |
| VST++-B |  Swin-B    |  [[baidu](https://pan.baidu.com/s/1cpuOj6jIwdld57UkDGMSfg?pwd=5t0v),PIN:5t0v]/ [[Geogle Drive](https://drive.google.com/file/d/1JhOFgsxlSWhhgOvvGzhHl2HcpJhuD_TG/view?usp=sharing)]  | [[baidu](),PIN:]/ [[Geogle Drive]()]|

### 2. Prediction Maps
We offer the prediction maps of **RGB** task [[baidu](https://pan.baidu.com/s/1KWLpliMzXDyOP7OmtmFVoA?pwd=e5sl),PIN:e5sl]/ [[Geogle Drive](https://drive.google.com/drive/folders/1ll3cALBffbt9FOKBtY-7UOxcYfFcsMUM?usp=sharing)] and **RGB-D** task [[baidu](),PIN:]/ [[Geogle Drive]()] at this time for all backbones.

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

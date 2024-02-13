# Video-SwinUNet
This is the official repo for the paper **[Training Dynamics in Physics-Informed Neural Networks with Feature Mapping](https://arxiv.org/abs/2402.06955)**.<br/>
Chengxi Zeng, Xinyu Yang, David Smithard, Majid Mirmehdi, Alberto M Gambaruto, Tilo Burghardt<br/>
<img src="https://github.com/SimonZeng7108/RBF-PINN/blob/master/Figs/lorenz.gif" width="500" height="280"><br/>

## Abstract
This paper presents a deep learning framework for medical video segmentation. Convolution neural network (CNN) and transformer-based methods have achieved great milestones in medical image segmentation tasks due to their incredible semantic feature encoding and global information comprehension abilities. However, most existing approaches ignore a salient aspect of medical video data - the temporal dimension. Our proposed framework explicitly extracts features from neighbouring frames across the temporal dimension and incorporates them with a temporal feature blender, which then tokenises the high-level spatio-temporal feature to form a strong global feature encoded via a Swin Transformer. The final segmentation results are produced via a UNet-like encoder-decoder architecture. Our model outperforms other approaches by a significant margin and improves the segmentation benchmarks on the VFSS2022 dataset, achieving a dice coefficient of 0.8986 and 0.8186 for the two datasets tested. Our studies also show the efficacy of the temporal feature blending scheme and cross-dataset transferability of learned capabilities.

## Architecture Overview
<img src="https://github.com/SimonZeng7108/Video-SwinUNet/blob/master/Figs/Video_swin.png" width="800" height="330"><br/>
(a)A ResNet-50 CNN feature extractor; (b)Temporal Context Module for temporal feature blending; (c)A Swin transformer-based feature encoder; (d)Cascaded CNN up-sampler for segmentation reconstruction; (e)2-layer segmentation head for detailed pixel-wise label mapping. Three skip connections are bridged between the CNN feature extractor and up-sampler as well as from the temporal features.<br/>

## Grad-Cam results & Snippet Size study
<p float="left">
  <img src="https://github.com/SimonZeng7108/Video-SwinUNet/blob/master/Figs/grad_cam_output.png" width="300" height="300" />
  <img src="https://github.com/SimonZeng7108/Video-SwinUNet/blob/master/Figs/confusion_matrix.png" width="335" height="305" /> 
</p>

Comparing the two closest competing architectures, grad-cam maps show where the model pays attention. Note the cleaner focus of our proposed approach.<br/>
Grid search over snippet sizes `t = 3, 5, 7, 9, 11 & 13` revealed the optimal, application-specific size `t = 5` both for training and testing.<br/>

## Qualitative results
**VFSS Part1**<br/>
<img src="https://github.com/SimonZeng7108/Video-SwinUNet/blob/master/Figs/quality.png" width="640" height="810"><br/>
**VFSS Part2**<br/>
<img src="https://github.com/SimonZeng7108/Video-SwinUNet/blob/master/Figs/qe_output.png" width="640"  height="510"><br/>

## Repo usage
### Requirements 
```Bash
conda create -n Video-SwinUNet python=3.8
conda activate Video-SwinUNet
pip install [following...]
```
- `torch == 1.10.1`
- `torchvision`
- `torchsummary`
- `numpy == 1.21.5`
- `scipy`
- `skimage`
- `matplotlib`
- `PIL`
- `mmcv == 1.5.0`
- `Medpy`
- `Timm`

### 1. Download pre-trained models
* [R50-ViT-B_16](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
* [Swin-T](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing)

### 2. Data 
Our data ethics approval only grants usage and showing on paper, not yet support full release. 
To fully utlise the Temporal Blending feature of the model, sequential image data should be converted to numpy arrays and concated in the format of `[T, H, W]` for BW data and `[T, C, H, W]` for colored data. Each .npz should have following structure: `{image: [T, H, W], label: [H, W], case_name:xxx}`

### 3. Train/Test
Train:<br/>
`python train.py --dataset Synapse --vit_name R50-ViT-B_16`<br/>
Test:<br/>
`python test_single.py --dataset Synapse --vit_name R50-ViT-B_16`<br/>

## Ref Repo
[Vision Transformer](https://github.com/google-research/vision_transformer)<br/>
[TransUNet](https://github.com/Beckschen/TransUNet/blob/main/README.md)<br/>
[SwinUNet](https://github.com/HuCaoFighting/Swin-Unet)<br/>
[Video-TransUNet](https://github.com/SimonZeng7108/Video-TransUNet)<br/>
[TCM](https://github.com/youshyee/Greatape-Detection)<br/>

## Citation
```
@inproceedings{Zeng2022VideoTransUNetTB,
  title={Video-TransUNet: temporally blended vision transformer for CT VFSS instance segmentation},
  author={Cheng Zeng and Xinyu Yang and Majid Mirmehdi and Alberto M. Gambaruto and Tilo Burghardt},
  booktitle={International Conference on Machine Vision},
  year={2022}
}
```


```
@misc{https://doi.org/10.48550/arxiv.2302.11325,
  doi = {10.48550/ARXIV.2302.11325},
  
  url = {https://arxiv.org/abs/2302.11325},
  
  author = {Zeng, Chengxi and Yang, Xinyu and Smithard, David and Mirmehdi, Majid and Gambaruto, Alberto M and Burghardt, Tilo},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Video-SwinUNet: Spatio-temporal Deep Learning Framework for VFSS Instance Segmentation},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```


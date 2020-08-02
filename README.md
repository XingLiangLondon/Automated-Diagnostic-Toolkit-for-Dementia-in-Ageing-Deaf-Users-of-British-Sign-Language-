# Automated Diagnostic Toolkit for Dementia in Ageing Deaf Users of British Sign Language
## Train and evaluate CNN deep learning model (ResNet50 / VGG16) based on transfer learning for early stage dementia screening in Keras

### Methodology
To train and evaluate CNN deep learning model ResNet50 / VGG16 based on transfer learning 
- Firstly please intall Keras deep learning model following the instructions of the link: https://github.com/fchollet/deep-learning-models. 
- Then download the code to use transfer learning in keras with the example of training a 2 class classification model using VGG-16 and Resnet-50 pre-trained weights. 


## Results
- ROC Curve and Confusion Matrix
<div align="center">
  <img src="Image/roc.png" alt="IMAGE ALT TEXT"></a>
</div>
<p align="center">
<img src="Image/cf_norm.png" width="300">        <img src="Image/cf_without_norm.png" width="300">
</p>  



## Citations
```
@inproceedings{liangECCV2020,
  author = {Xing Liang, Anastassia Angelopoulou, Epaminondas Kapetanios, Bencie Woll, Reda Al-batat and Tyron Woolfe},
  booktitle = {Sign Language Recognition, Translation & Production (SLRTP) Workshop, in conjunction with the ECCV 2020 Conference},
  title = {A Multi-modal Machine Learning Approach and Toolkit to Automate Recognition of Early Stages of Dementia among British Sign Language Users},
  year = {2020}
}
```

# Image-Captioning
Computer Vision Final Project

Member: Peilun Zhang, Yichen Gong

Download dataset from http://mscoco.org/dataset/#download

Here is how you get the project running

```
git clone https://github.com/plzhang/Image-Captioning.git  

sh download_dataset  

sh preprocess.sh  

th main.lua -cuda # If you use cuda  

```

If you want to use the model without the need to use entire VGG-19 for each forward. simply type
```
git checkout sep
```

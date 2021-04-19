# ScanGAN360
Code and model for the paper "[ScanGAN360: A Generative Model of Realistic Scanpaths for 360º Images](http://webdiis.unizar.es/~danims/projects/vr-scanpaths.html)".

![Teaser](https://github.com/DaniMS-ZGZ/ScanGAN360/blob/main/img/vr-scanpaths.jpg)

## Requirements
This work was developed using:
```
* python 3.7.4
* pytorch 1.2.0
* cudatoolkit 10.0.30
* opencv 4.1.2
```

## Inference
The current version of the repository includes a basic, yet functional version to generate scanpaths from a 360º image using the ScanGAN360 model.

### Usage
There is currently one mode of usage for this code:
```
python main.py --mode inference 
```

This will read an image `image_path = "data/test.jpg"` and generate a set of scanpaths that will be saved in `path_to_save = "test/"`. You can modify both those paths, and the number of generated scanpaths `n_generated`. Each of the images will contain 25 different scanpaths.

## Training the model
This option will be available soon.

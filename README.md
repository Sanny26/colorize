# Colorize

Builds a model to automatically color a grayscale image.

To train the model:
* Create directories `input/` and `output/`. In `input/` folder, grayscale images that need to be trained should be stored. In `output/` folder, color images for the respective grayscale image in input/ folder should be stored, this image should have the same name as the grayscale image.
* When running the model for the first time, use `python3 clrmdl.py` 
* When retraining an already saved model, use `python3 clrmdl.py retrain`

Model related information will be stored in `model.h5` and `model.json`

Model inspiration from [here](http://tinyclouds.org/colorize/)




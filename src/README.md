# Artifacts 

This folder contains all the code we used for models, training, processing, finetuning and our ensemble method.


### base_model.ipynb
- Used for loading the data and model, training the model and storing the model.

### ensemble.ipynb
- Used for evaluation of each model on every type of question class.

### finetune.py
- Used for finetuning all layers of a selected pretrained architecture from HuggingFace.

### model.py
- Base architecture for our transfer learning models, actual architecture used is in folder [./custom_models].



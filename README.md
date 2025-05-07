# Robot Foveas

## Guide to the directories

# FovConvNeXt
The code from Killick et al, 2023. This folder contains the code required to create a foveated graph neural network and a uniform, unfoveated one (when fovea radius is set to 0). We access it by calling `from FovConvNeXt.models import make_model`

# foveated_training_imagenet100
This folder contains the best version of the foveated model when it was trained on Imagenet 100. It also contains a plaintext file with the output of the training, detailing the train & test accuracy & loss at each epoch. Checkpoints were saved every 10 epochs, but not included in the repo in the interest of cloud storage space. The model in this folder was trained by running train.py (a notebook version of which is included with train.ipynb). The unfoveated model was trained with the same function, but had radius=0 instead of radius=0.4.

# jetbot_dataset (in the form of jetbot_dataset.zip)
This folder contains all the images taken on the jetbot, before the camera stopped working. It contains roughly 100 blocked and 100 free images, each in their own folder. These images were used to fine tune the models. They were taken using jetbot_data_collection.ipynb and the models were fine tuned using transfer_learning.ipynb.

# deprecated
This folder contains all our old code or random files that got used transiently (so we want to save them), but ultimately are not important for result replication.

## Files

RACE.ipynb - designed to be run on the Jetson Nano, this notebook compares the speeds of the fine tuned models (stored in the fine_tuned_*.pth files). It likely will not work outside of the Nano due to jetbot-specific optimizations (namely, the conversion of all numbers to torch.half()).

stats_tests.py - this script performs statistical analyses on the output of RACE.ipynb.

fine_tuning_graph.py - produced lineplots of the accuracy vs epoch during model fine tuning

fovea_shape_display.ipynb - this displays the shape of the foveated and unfoveated sampling patterns.

jetbot_data_collection, train.py/train.ipynb, and fine_tuning.ipynb - these were all explained in the directories section.

*_output.txt - these represent the various text outputs of the script included in their name for the condition included in their name.
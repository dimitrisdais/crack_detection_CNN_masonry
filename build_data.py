"""
The following code was produced for the Journal paper 
"Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning"
by D. Dais, İ. E. Bal, E. Smyrou, and V. Sarhosis published in "Automation in Construction"
in order to apply Deep Learning and Computer Vision with Python for crack detection on masonry surfaces.

In case you use or find interesting our work please cite the following Journal publication:

D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces 
using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. 
https://doi.org/10.1016/j.autcon.2021.103606.

@article{Dais2021,
author = {Dais, Dimitris and Bal, İhsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},
doi = {10.1016/j.autcon.2021.103606},
journal = {Automation in Construction},
pages = {103606},
title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},
url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},
volume = {125},
year = {2021}
}

The paper can be downloaded from the following links:
https://doi.org/10.1016/j.autcon.2021.103606
https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning/stats

The code used for the publication can be found in the GitHb Repository:
https://github.com/dimitrisdais/crack_detection_CNN_masonry

Author and Moderator of the Repository: Dimitris Dais

For further information please follow me in the below links
LinkedIn: https://www.linkedin.com/in/dimitris-dais/
Email: d.dais@pl.hanze.nl
ResearchGate: https://www.researchgate.net/profile/Dimitris_Dais2
Research Group Page: https://www.linkedin.com/company/earthquake-resistant-structures-promising-groningen
YouTube Channel: https://www.youtube.com/channel/UCuSdAarhISVQzV2GhxaErsg  

Your feedback is welcome. Feel free to reach out to explore any options for collaboration.
"""

import os

folder = {}
# Use this to easily run the code in different directories/devices
folder['initial'] = 'C:/Users/jimar/Dimitris/python/'
# The path where the repository is stored
folder['main'] = folder['initial'] + 'crack_detection_CNN_masonry/'

# if folder['main'] == '', then the current working directory will be used
if folder['main'] == '':
    folder['main'] = os.getcwd()

import sys
sys.path.append(folder["main"])

from config_class import Config

cnf = Config(folder["main"])
args = cnf.set_repository()

# Set some parameters
IMAGE_DIMS = cnf.IMAGE_DIMS

# import the necessary packages
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from imutils import paths
import numpy as np
import progressbar
import cv2

from subroutines.HDF5 import HDF5DatasetWriterMask

# grab the paths to the images and masks
trainPaths = list(paths.list_images(args['images']))
trainLabels = list(paths.list_images(args['masks'])) 

# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels,
                         test_size=cnf.TEST_SIZE, random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
    ("train", trainPaths, trainLabels, args['TRAIN_HDF5']), 
    ("val", valPaths, valLabels, args['VAL_HDF5'])]


# loop over the dataset tuples
for (dType, images_path, masks_path, outputPath) in datasets:
    
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriterMask((len(images_path), IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]), outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(images_path),
                                   widgets=widgets).start()
    
    # loop over the image paths
    for (ii, (im_path, mask_path)) in enumerate(zip(images_path, masks_path)):
        
        # load the image and process it
        image = cv2.imread(im_path)
        
        # resize image if dimensions are different
        if IMAGE_DIMS != image.shape:
            image = resize(image, (IMAGE_DIMS), mode='constant', preserve_range=True)
        
        # normalize intensity values: [0,1]
        image = image / 255
        
        # label
        mask = cv2.imread(mask_path, 0)
        
        # resize image if dimensions are different
        if IMAGE_DIMS[0:2] != mask.shape:
            mask = resize(mask, (IMAGE_DIMS[0], IMAGE_DIMS[1]), mode='constant', preserve_range=True)

        # normalize intensity values: [0,1]            
        mask = np.expand_dims(mask, axis=-1)
        mask = mask / 255

        # add the image and label to the HDF5 dataset
        writer.add([image], [mask])
        
        # update progress bar
        pbar.update(ii)
    
    # close the progress bar and the HDF5 writer
    pbar.finish()
    writer.close()

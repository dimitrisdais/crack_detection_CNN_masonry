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

"""
The hdf5datasetgenerator_mask.py is based on material provided by Adrian Rosebrock shared on his blog (https://www.pyimagesearch.com/) and his books
The original code was prepared for classification while here it has been adjusted to work for segmentation; 
the ground truth masks are the labels passed along with each image

Adrian Rosebrock, Deep Learning for Computer Vision with Python - Practitioner Bundle, 
    PyImageSearch, https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/, 
    accessed on 24 February 2021

Adrian Rosebrock, How to use Keras fit and fit_generator (a hands-on tutorial), 
    PyImageSearch, https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/, 
    accessed on 24 February 2021
"""
    
# import the necessary packages
import numpy as np
import h5py

class HDF5DatasetGeneratorMask:
    def __init__(self, dbPath, batchSize, preprocessors=None, shuffle=False,
                 aug=None, binarize=True, classes=2, threshold=0.5):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.shuffle = shuffle
        
        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r+')
        self.numImages = self.db["labels"].shape[0]
        # threshold to binarize mask
        self.threshold = threshold
        
    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]    
                
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        procImages.append(image)

                    # update the images array to be the processed
                    # images
                    images = np.array(procImages)                
                
                # if the data augmenator exists, apply it
                if self.aug is not None:
                    
                    # perform the same transformation to image and mask
                    seed = 2018
                    
                    image_generator = self.aug.flow(images, seed=seed, batch_size=self.batchSize, shuffle=self.shuffle)
                    mask_generator = self.aug.flow(labels, seed=seed, batch_size=self.batchSize, shuffle=self.shuffle)

                    train_generator = zip(image_generator, mask_generator)
                    (images, labels) = next(train_generator)
                    
                    # process labels so that they have only 0 or 1
                    if self.binarize:
                        for ii in range(0, len(labels)):
                            labels[ii] = np.where(labels[ii]>self.threshold, 1., 0.)
        
                # yield a tuple of images and labels
                yield (images, labels)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()  
		

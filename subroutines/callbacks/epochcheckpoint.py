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
The epochcheckpoint.py is based on material provided by Adrian Rosebrock shared on his blog (https://www.pyimagesearch.com/) and his books

Adrian Rosebrock, Deep Learning for Computer Vision with Python - Practitioner Bundle, 
    PyImageSearch, https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/, 
    accessed on 24 February 2021

Adrian Rosebrock, Keras: Starting, stopping, and resuming training, 
    PyImageSearch, https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/, 
    accessed on 24 February 2021
"""

# import the necessary packages
from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath_checkpoints, outputPath_weights, save_model_weights,
              every=5, startAt=0, info = '', counter = '', extension = '.h5'):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath_checkpoints = outputPath_checkpoints
		self.outputPath_weights = outputPath_weights
		self.save_model_weights = save_model_weights
		self.every = every
		self.intEpoch = startAt
		self.info = info
		self.counter = counter
		self.extension = extension

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
            
			# check whether to save the whole model or only the weight
			if self.save_model_weights == 'model':
				folder_output = self.outputPath_checkpoints
                
			elif self.save_model_weights == 'weights':
				folder_output = self.outputPath_weights
				
            # define the name of saved model/weights
			if self.info=='':
			    p = os.path.sep.join([folder_output,
    				"{}_epoch_{}{}".format(self.counter, self.intEpoch + 1, self.extension)])
			else:
			    p = os.path.sep.join([folder_output,
    				"{}_{}_epoch_{}{}".format(self.info, self.counter, self.intEpoch + 1, self.extension)])

			# check whether to save the whole model or only the weight
			if self.save_model_weights == 'model':
			    self.model.save(p, overwrite=True)
            
			elif self.save_model_weights == 'weights':                
			    self.model.save_weights(p)

		# increment the internal epoch counter
		self.intEpoch += 1
        

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

import sys
from keras.models import model_from_json

class LoadModel:
    def __init__(self, args, IMAGE_DIMS, BS):
        self.args = args
        self.IMAGE_DIMS = IMAGE_DIMS
        self.BS = BS

    def load_pretrained_model(self):
        """
        Load a pretrained model
        """
        
        # Load pretrained DeepCrack
        if self.args["model"] == 'DeepCrack':
            
            sys.path.append(self.args["main"] + 'networks/')
            from edeepcrack_cls import Deepcrack

            model = Deepcrack(input_shape=(self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))

            # load weights into new model
            model.load_weights(self.args['weights'] + self.args['pretrained_filename'])

        # Load pretrained model
        # This option is not supported for the current version of the code for the 'evaluation' mode
        # Print an explanatory comment and exit
        elif self.args['save_model_weights'] == 'model':
            
            raise ValueError("The option to load a model is not supported for the 'evaluation' mode." +
                  "In case you need to use the pretraine model to perform predictions, then" +
                  "train the model with the option: args['save_model_weights'] == 'weights'" +
                  "\nThe analysis will be terminated")
                
        # Load model from JSON file and then load pretrained weights
        else:
            
            # If pretrained Deeplabv3 will be loaded, import the Deeplabv3 module
            if self.args["model"] == 'Deeplabv3':            
                sys.path.append(self.args["main"] + 'networks/')
                from model import Deeplabv3  

            # load json and create model
            json_file = open(self.args['model_json'], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            try:
                model = model_from_json(loaded_model_json)
            except:
                from tensorflow.keras.models import model_from_json
                model = model_from_json(loaded_model_json)
        
            # load weights into new model
            model.load_weights(self.args['weights'] + self.args['pretrained_filename'])
        
        return model

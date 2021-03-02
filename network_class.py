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

class Network:
    def __init__(self, args, IMAGE_DIMS, N_FILTERS, BS, INIT_LR, opt, loss, metrics):
        self.args = args
        self.IMAGE_DIMS = IMAGE_DIMS
        self.N_FILTERS = N_FILTERS
        self.BS = BS
        self.INIT_LR = INIT_LR
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

    def add_regularization_function(self, args, model):
        """ 
        If args['regularization'] is not None, add regularization and then return the model
        """
        if self.args['regularization'] != None:
            from networks import add_regularization
            model = add_regularization(model, self.args['regularization'])
        
        return model
            
    def define_Network(self): 
        
        sys.path.append(self.args["main"])

        if self.args["model"] == 'Unet':
            
            from networks import Unet

            model = Unet(self.IMAGE_DIMS, n_filters=self.N_FILTERS, dropout=self.args['dropout'], 
                             batchnorm=self.args['batchnorm'], regularization=self.args['regularization'], 
                             kernel_initializer=self.args['init'])   

        elif 'sm' in self.args["model"]:
    
            # Refer to the following GitHub repository for the implementation of the Segmentation Models
            # with pretrained backbones
            # https://github.com/qubvel/segmentation_models  
            import segmentation_models as sm    
            
            _, model_to_use, BACKBONE = self.args["model"].split('_') # ['sm', 'FPN', 'mobilenet']

            # Define network parameters
            n_classes = 1
            activation = 'sigmoid'
            encoder_weights= self.args['encoder_weights'] # None or 'imagenet'
    
            # Define model    
            if model_to_use == 'FPN':
                pyramid_block_filters=512
                model = sm.FPN(BACKBONE, input_shape=self.IMAGE_DIMS, classes=n_classes, activation=activation, encoder_weights=encoder_weights,
                                pyramid_block_filters=pyramid_block_filters, pyramid_dropout = self.args['dropout'])
                   
            elif model_to_use == 'Unet':
                model = sm.Unet(BACKBONE, input_shape=self.IMAGE_DIMS, classes=n_classes, activation=activation, encoder_weights=encoder_weights,
                                decoder_filters=(1024, 512, 256, 128, 64), dropout = self.args['dropout'])
                
            # If requested, add regularization and then return the model
            model = self.add_regularization_function(self.args, model)
            
        elif self.args["model"] == 'Deeplabv3':
            
            # Refer to the following GitHub repository for the implementation of DeepLab 
            # https://github.com/tensorflow/models/tree/master/research/deeplab
            
            sys.path.append(self.args["main"] + 'networks/')
            from model import Deeplabv3        
            
            weights='pascal_voc'
            input_shape=self.IMAGE_DIMS
            classes = 1
            BACKBONE = 'xception' # 'xception','mobilenetv2'
            activation = 'sigmoid'# One of 'softmax', 'sigmoid' or None
            OS=16 # {8,16}
        
            model = Deeplabv3(weights=weights, input_shape=input_shape, classes=classes, backbone=BACKBONE,
                      OS=OS, activation=activation)
            
            import tensorflow as tf
            self.opt = tf.keras.optimizers.Adam(self.INIT_LR)

        elif self.args["model"] == 'DeepCrack':      

            # Refer to the following GitHub repository for the implementation of DeepCrack 
            # https://github.com/hanshenChen/crack-detection

            sys.path.append(self.args["main"] + 'networks/')
            from edeepcrack_cls import Deepcrack

            model = Deepcrack(input_shape=(self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))

            import tensorflow as tf
            self.opt = tf.keras.optimizers.Adam(self.INIT_LR)
   
        model.compile(optimizer=self.opt, loss=self.loss, metrics=[self.metrics])
        
        return model

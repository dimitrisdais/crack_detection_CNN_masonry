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

class Loss:
    def __init__(self, args):
        self.args = args
        
    def define_Loss(self): 
        
        sys.path.append(self.args["main"])

        # choose loss
        if self.args['loss'] == 'Focal_Loss':
            from subroutines.loss_metric import Focal_Loss
            loss = Focal_Loss(alpha=self.args['focal_loss_a'], gamma=self.args['focal_loss_g'])
            
        elif self.args['loss'] == 'WCE':
            from subroutines.loss_metric import Weighted_Cross_Entropy
            loss = Weighted_Cross_Entropy(beta=self.args['WCE_beta'])
            

        elif self.args['loss'] == 'F1_score_Loss':
            from subroutines.loss_metric import F1_score_Loss
            loss = F1_score_Loss 
            
        elif self.args['loss'] == 'F1_score_Loss_dil':
            from subroutines.loss_metric import F1_score_Loss_dil
            loss = F1_score_Loss_dil  
            
        elif self.args['loss'] == 'Binary_Crossentropy':
            import tensorflow as tf
            loss=tf.keras.losses.BinaryCrossentropy()
            
        return loss
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
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 

def Visualize_Predictions(args, predictions, threshold=0.5):
    """
    threshold: use this value to binarize the ground trough mask
    """
    
    sys.path.append(args["main"])
    
    from subroutines.loss_metric import DilateMask
    from subroutines.loss_metric import Recall_np
    from subroutines.loss_metric import Precision_np
    from subroutines.loss_metric import F1_score_np
     
    # Load database with images and masks
    db = h5py.File(args['EVAL_HDF5'], 'r+')
    numImages = db["images"].shape[0]    

    # Show the crack as grey    
    color = 0.5 # 0.5:grey, 1: white
     
    # Size of plot
    plt_size = [3, 3]
    
    # Loop over the images and produce a plot with original image, ground truth and prediction
    for ii in range (0, numImages):
        
        # Filename to store plot for each image
        plt_file = '{}{}.png'.format(args['predictions_subfolder'], ii)

        # Ground truth
        gt = (db["labels"][ii].squeeze())*color
        
        # Define im array
        im = (np.zeros(db["images"][ii].shape)).astype('uint8') 
        im[:,:,0] = (db["images"][ii][:,:,2] * 255) .astype('uint8') 
        im[:,:,1] = (db["images"][ii][:,:,1] * 255) .astype('uint8') 
        im[:,:,2] = (db["images"][ii][:,:,0] * 255) .astype('uint8') 
    
        # Check whether to dilate ground truth mask for the calculation of Precision metric
        if args['predictions_dilate']:
            y_true_Precision = DilateMask(db["labels"][ii])
        else:
            y_true_Precision = db["labels"][ii]
        
        # Calculate metrics for each image
        recall = Recall_np(db["labels"][ii], predictions[ii])
        precision = Precision_np(y_true_Precision, predictions[ii])
        f1_score = F1_score_np(recall, precision)
        # Format metrics
        recall = int(round(recall, 2)*100)
        precision = int(round(precision, 2)*100)
        f1_score = int(round(f1_score, 2)*100)
        
        plt_subtitle = 'F1:{0}% / RE:{1}%\nPR:{2}%'.format(f1_score, recall, precision)
    
        # Create figure with 3 subplots
        fig = plt.figure()
        fig.set_size_inches(plt_size)
        ax1 = plt.subplot2grid((1,1), (0,0))
        divider = make_axes_locatable(ax1) 
        ax2 = divider.append_axes("bottom", size="100%", pad=0.1)
        ax3 = divider.append_axes("bottom", size="100%", pad=0.4)
        
        # Original image
        ax1.imshow(im)
        # Ground truth
        ax2.imshow(gt, vmin=0, vmax=1, cmap='gray')
        # Prediction     
        prediction = ((predictions[ii].squeeze()>threshold)*1)*color 
        ax3.imshow(prediction, vmin=0, vmax=1, cmap='gray')
        
        # Set title for prediction
        ax3.set_title(plt_subtitle, fontsize=7)  
        
        # Remove axes
        ax1.axis('off')  
        ax2.axis('off')  
        ax3.axis('off') 
    
        plt.tight_layout()
        plt.savefig(plt_file, bbox_inches = "tight", dpi=100, pad_inches=0.05) 
        plt.close()
        
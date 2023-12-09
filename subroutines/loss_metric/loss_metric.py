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

#%%

import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2

#%%
# Function to dilate mask
# It is used to dilate the y_true when calculating precision in order to allow for some tolerance
# 
# Background pixels predicted as cracks (False Positives) were considered as True Positives 
# if they were a few pixels apart from the annotated cracks
#

def dilation2d(img4D):

    # The greater the kernel size, the greater extent of the dilation applied
    kernel_size = 5
    
    with tf.compat.v1.variable_scope('dilation2d'):
        kernel = tf.zeros((kernel_size, kernel_size, 1)) 
        output4D = tf.nn.dilation2d(img4D, filter=kernel, strides=(1,1,1,1), data_format='NHWC', dilations=(1,1,1,1), padding="SAME")

        return output4D

#%%
# Weighted Cross-Entropy (WCE) Loss
# It is based on the implementation found in the link below:
# https://jeune-research.tistory.com/entry/Loss-Functions-for-Image-Segmentation-Distribution-Based-Losses
#

def Weighted_Cross_Entropy(beta):
  def convert_to_logits(y_pred):
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)

    return tf.reduce_mean(loss)

  return loss

#%%
# Focal Loss
# It is based on the implementation found in the link below:
# https://github.com/umbertogriffo/focal-loss-keras
#

def Focal_Loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # Clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

#%%
# F1-score Loss
#

def F1_score_Loss(y_true, y_pred):

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

#%%
# F1-score Loss with dilated y_true mask
#

def F1_score_Loss_dil(y_true, y_pred):
    
    smooth = 1.
    # Dilate y_true
    y_true_dil = dilation2d(y_true)
    y_true_dil = K.flatten(y_true_dil)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_dil * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

#%%
# Recall Metric
#

def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#%%
# Precision Metric
#

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#%%
# Precision Metric with dilated y_true mask
#

def Precision_dil(y_true, y_pred):
    
    # Dilate y_true
    y_true = dilation2d(y_true)
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#%%
# F1-score Metric
#

def F1_score(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%%
# F1-score Metric with dilated y_true mask
#

def F1_score_dil(y_true, y_pred):
    precision = Precision_dil(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#%%
# Define metrics to be used for evaluation of the trained model using NumPy instead of tensorflow tensors
#

def DilateMask(mask, threshold=0.5, iterations=1):
    """
    receives mask and returns dilated mask
    """

    kernel = np.ones((5,5),np.uint8)
    mask_dilated = mask.copy()
    mask_dilated = cv2.dilate(mask_dilated,kernel,iterations = iterations)
    # Binarize mask after dilation
    mask_dilated = np.where(mask_dilated>threshold, 1., 0.)  

    return mask_dilated
	
def Recall_np(y_true, y_pred, threshold=0.5):
    
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(),half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    possible_positives = y_true_f.sum()
    recall = (true_positives + eps) / (possible_positives + eps)
    return recall

def Precision_np(y_true, y_pred, threshold=0.5):
    
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(),half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    predicted_positives = y_pred_f.sum()
    precision = (true_positives + eps) / (predicted_positives + eps)
    
    return precision

def F1_score_np(recall,precision):
   
    eps = 1e-07
    return 2*((precision*recall)/(precision+recall+eps))

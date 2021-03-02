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
import sys
  
class Config:
    def __init__(self, working_folder):
        
        # The path where the repository is stored
        self.working_folder = working_folder
        
        # Define the mode that will be used when running the code
        self.mode = 'build_data' # 'train', 'evaluate' or 'build_data'
        # Info that will be used as prefix of any output files
        self.info = 'crack_detection'
        # Dimensions of the images that will be fed to the network
        self.IMAGE_DIMS = (224 , 224 , 3)
        # Batch size
        self.BS = 4
        # Number of epochs to train the model
        self.epochs = 10
        # Initial learning rate
        self.INIT_LR = 0.0005
        
        # The parameters of the configuration used will be stored in the dictionary args
        self.args = {}
        
        # 1) 'Unet'
        # 2) 'Deeplabv3': https://github.com/tensorflow/models/tree/master/research/deeplab
        # 3) 'DeepCrack': https://github.com/hanshenChen/crack-detection
        # 4) Different segmentation models as defined by https://github.com/qubvel/segmentation_models
        #    The name of the model should be structured as follows: sm_ModelName_Backbone
        #    e.g. 'sm_Unet_mobilenet': Unet will be used as model and mobilenetv2 as backbone
        #    The acceptable ModelNames are Unet and FPN
        #    For the acceptable Backbone networks refer to the documentation of the GitHub repository
        #    https://github.com/qubvel/segmentation_models#models-and-backbones
        self.args['model'] = 'Unet' # 'Deeplabv3', 'Unet', 'DeepCrack', 'sm_Unet_mobilenet'
        
        # Define regularization for Unet and SM networks
        self.args['regularization'] = 0.001 # regularization: None or insert a value (e.g. 0.0001)
        # define optimizer
        self.args['opt'] = 'Adam' # 'SGD' or 'Adam' or 'RMSprop'
        # define whether data augmentation will be used
        self.args['aug'] = False # True or False
        # define whether dropout will be used for Unet and SM networks
        self.args['dropout'] = None # dropout: None or insert a value (e.g. 0.5)
        # define whether batch normalization  will be used for Unet         
        self.args['batchnorm'] = True # True or False
        
        # Define whether to save the whole model or only the weights
        #
        # If 'model' is chosen, the model will be saved in the folder checkpoints
        # using the the command: model.save(filepath)
        # The saved model contains: 
        #   the model's configuration (topology)
        #   the model's weights
        #   the model's optimizer's state (if any)
        #
        # If 'weights' is chosen, only the layer weights of the model will be saved
        # in the folder weights using the the command: model.save_weights(filepath) 
        #
        # For this version of the code only the option 'weights' is supported
        # In a later version the option 'model' will be supported as well
        self.args['save_model_weights'] = 'weights' # 'model' or 'weights'
        
        # Parameters to define for the configuration of Unet
        # Number of filters for Unet
        self.N_FILTERS = 64
        # Define the Layer weight initializers for Unet
        self.args['init'] = 'he_normal' # kernel_initializer: 'he_normal' or 'glorot_uniform' or 'random_uniform'
        
        # Parameters to define for the configuration of the SM networks
        # if 'imagenet' is used the backbone network will be initialized with
        # weights pretrained on ImageNet
        self.args['encoder_weights'] = 'imagenet' # None or 'imagenet' 
        
        # Define Loss Function
        # 'Focal_Loss'
        # 'WCE': Weighted Cross-Entropy
        # 'BCE': Binary Cross-Entropy
        # 'F1_score_Loss'
        # 'F1_score_Loss_dil': Background pixels predicted as cracks (FP) are considered as
        #                      TP if they are a few pixels apart from the annotated cracks
        #                      Refer to the Journal paper for extra clarification
        self.args['loss'] = 'WCE'
        # define alpha and gamma values only when focal loss is used
        if self.args['loss'] == 'Focal_Loss':
            self.args['focal_loss_a'] = 0.25 # alpha value when focal loss is used default: 0.25
            self.args['focal_loss_g'] = 2 # gamma value when focal loss is used. default: 2.
        # Define WCE_beta only when WCE loss is used            
        elif self.args['loss'] == 'WCE': # Weighted Cross-Entropy
            # For extra documentation regarding 'WCE_beta' see the following link
            # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
            self.args['WCE_beta'] = 10 # weight of the positive class (i.e. crack class)
        
        # See TrainingMonitor for extra details
        self.args['start_epoch'] = 0
        self.args['every'] = 5
 
        # Metric to be used for TrainingMonitor and ModelCheckpoint 
        self.args['metric_to_plot'] = 'F1_score_dil'
        
        # Whether data generators will binarize masks 
        self.args['binarize'] = True # True or False
        
        # When building the HDF5 database from the images of the dataset
        # the dataset will be split into train and validation
        # define the portion (0 to 1) of the whole dataset that will be used for validation
        self.TEST_SIZE = 0.40
        
    def check_folder_exists(self, folder_path):
        """ 
        check if folder exists and if not create it
        """
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def set_repository(self):
        
        # Pass the path where the repository is stored to the args
        self.args['main'] = self.working_folder
        # Where the dataset is stored
        self.args['dataset'] = self.args['main'] + 'dataset/'
        # A new folder will be created where any output will be stored in
        self.args['output'] = self.args['main'] + 'output/'
        # The dataset will be stored in HDF5 format here
        self.args['hdf5'] = self.args['output'] + 'hdf5/'
        # The saved model will be stored here
        self.args['checkpoints'] = self.args['output'] + 'checkpoints/'
        # The saved weights will be stored here        
        self.args['weights'] = self.args['output'] + 'weights/'
        # Theserialized model (JSON) will be stored here
        self.args['model_json_folder'] = self.args['output'] + 'model_json/'
        # Predictions will be stored here
        self.args['predictions'] = self.args['output'] + 'predictions/'
        
        # Create the folders
        folders = [self.args['hdf5'], self.args['checkpoints'], self.args['weights'], 
                   self.args['model_json_folder'], self.args['predictions']]
        for f in (folders):
            self.check_folder_exists(f)

        # Save the HDF5 file to different folder according to IMAGE_DIMS
        temp = '{}_{}_{}_{}/'.format(self.info, self.IMAGE_DIMS[0],self.IMAGE_DIMS[1],self.IMAGE_DIMS[2])
        self.check_folder_exists(self.args['hdf5'] + temp)
        # Define the output for the train and validation HDF5 files
        self.args['TRAIN_HDF5'] = self.args['hdf5'] + temp + 'train.hdf5'
        self.args['VAL_HDF5'] = self.args['hdf5'] + temp + 'val.hdf5'
        # In case you need to test the model on a set other than the validation set,
        # define the EVAL_HDF5 suitably
        self.args['EVAL_HDF5'] = self.args['hdf5'] + temp + 'val.hdf5'
            
        # Define the path that the patches of images and masks are stored
        self.args['images'] = self.args['dataset'] + '{}_{}_images/'.format(self.info, self.IMAGE_DIMS[0])
        self.args['masks'] = self.args['dataset'] +'{}_{}_masks/'.format(self.info, self.IMAGE_DIMS[0])
    
        # Diffent configurations when mode is 'train' or evaluate'
        if self.mode == 'train':

            # When running different trials, the output files will not overwrite the existing ones
            # The output files of each trial will have a different suffix
            #
            # Check in the output folder if there is a  file named 'counter.txt'
            # The file should contain only a number that will be used as counter 
            # If the counter file doesn't exist, use the os.getpid() as counter
            
            # Path to the counter file
            self.args['counter_file'] = self.args['output'] + 'counter.txt'
            
            # If 'counter.txt' exists read the counter
            if os.path.exists(self.args['counter_file']):
                # Ask whether to change counter. acceptable answers are 'y' or 'n'
                # If anything else is given as input, ask again
                while True:
                    self.args['counter_check'] = input('Shall I change the counter [y/n]:')
                    if self.args['counter_check'] == 'y' or self.args['counter_check'] == 'n':
                        break
                    else:
                        continue
                
                # If input was 'y', change the counter   
                if self.args['counter_check'] == 'y':
                    file = open(self.args['counter_file'], 'r') 
                    self.args['counter'] = int(file.read())+1
                    file.close()
                    file = open(self.args['counter_file'], 'w') 
                    file.write(str(self.args['counter'])) 
                    file.close()
                # If input was 'n', use the same counter   
                else:
                    file = open(self.args['counter_file'], 'r') 
                    self.args['counter'] = int(file.read())
                    file.close()
                    
            # If the counter file doesn't exist, use the os.getpid() as counter
            else:
                self.args['counter'] = os.getpid()
            
            # Print the counter
            print(self.args['counter'])
            
            # Store results (i.e. metrics, loss) to CSV format
            # Check if the counter value was used before
            # If it was used before ask the user whether to proceed
            # If 'n' is passed, the analysis will be terminated
            self.args['CSV_PATH'] = self.args['output'] + '{}_{}.out'.format(self.info, self.args['counter'])
            if os.path.exists(self.args['CSV_PATH']):
                print("The counter '{}' has been used before\nShould the analysis continue [y/n]:".format(self.args['counter']))
                check = input()
                if check == 'n':
                    print('The analysis will be terminated')
                    sys.exit(1)
            
            # Plot Loss/Metrics during training
            self.args['FIG_PATH'] = self.args['output'] + self.info + '_{}.png'.format(self.args['counter'])
            # Serialize results (i.e. metrics, loss) to JSON
            # If None is given, no serialization will take place
            self.args['JSON_PATH'] = self.args['output'] + self.info + '_{}.json'.format(self.args['counter'])
            # Plot the architecture of the network
            self.args['architecture'] = self.args['output'] + '{}_architecture_{}.png'.format(self.info, self.args['counter'])
            # Store the summary of the model to txt file
            self.args['summary'] = self.args['output'] + '{}_summary_{}.txt'.format(self.info,self.args['counter'])        
            
        elif self.mode == 'evaluate':
            
            # Define the counter suitably in order to read the correct JSON file etc.
            self.args['counter'] = 6
            # Define the file with the pretrained weights or the model with weights that will be used to evaluate model
            # e.g. 'crack detection_1_epoch_7_F1_score_dil_0.762.h5'
            self.args['pretrained_filename'] = 'crack_detection_6_epoch_9_F1_score_dil_0.809.h5'
            # Define the subfolder where predictions will be stored
            self.args['predictions_subfolder'] = '{}{}/'.format(self.args['predictions'], self.args['pretrained_filename'])
            # Define whether to dilate ground truth mask for the calculation of Precision metric
            # Background pixels predicted as cracks (FP) are considered as TP if they are a few 
            # pixels apart from the annotated cracks. Refer to the Journal paper for extra clarification                 
            self.args['predictions_dilate'] = True # True or False
            
        # Configurations to be used both for 'training' and 'evaluation' modes
        if (self.mode == 'train') or (self.mode == 'evaluate'):
            
            # The path for the serialized model to JSON
            self.args['model_json'] = self.args['model_json_folder'] + self.info + '_{}.json'.format(self.args['counter']) 
        
        return self.args
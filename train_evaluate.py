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
sys.path.append(folder['main'])

from config_class import Config

cnf = Config(folder['main'])
args = cnf.set_repository()

# Set some parameters
IMAGE_DIMS = cnf.IMAGE_DIMS
BS = cnf.BS
epochs = cnf.epochs
INIT_LR = cnf.INIT_LR
N_FILTERS = cnf.N_FILTERS
info = cnf.info
mode = cnf.mode

# When using DeepCrack, eager execution needs to be enabled
if args["model"] == 'DeepCrack':
    import tensorflow as tf
    tf.enable_eager_execution()

from subroutines.HDF5 import HDF5DatasetGeneratorMask

#%%
  
if mode == 'train':

    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import CSVLogger
    
    from subroutines.callbacks import EpochCheckpoint
    from subroutines.callbacks import TrainingMonitor
    from subroutines.visualize_model import visualize_model
    
    #%%  
    # Prepare model for training
    #
    
    # Define metrics
    from metrics_class import Metrics
    metrics = Metrics(args).define_Metrics()
    
    # Define loss
    from loss_class import Loss
    loss = Loss(args).define_Loss()
    
    # Define optimizer
    from optimizer_class import Optimizer
    opt = Optimizer(args, INIT_LR).define_Optimizer()
    
    # Define Network and compile model
    from network_class import Network
    model = Network(args, IMAGE_DIMS, N_FILTERS, BS, INIT_LR, opt, loss, metrics).define_Network()

    # Visualize model
    try:
        visualize_model(model, args['architecture'], args['summary'])
    except:
        from subroutines.visualize_model import visualize_model_tf
        visualize_model_tf(model, args['architecture'], args['summary'])
    
    #%%
        
    # Data augmentation for training and validation sets
    if args['aug'] == True:
        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode='nearest')
    else:
        aug = None
    
    # Load data generators
    trainGen = HDF5DatasetGeneratorMask(args['TRAIN_HDF5'], BS, aug=aug, shuffle=False, binarize=args['binarize'])
    valGen = HDF5DatasetGeneratorMask(args['VAL_HDF5'], BS, aug=aug, shuffle=False, binarize=args['binarize'])

    #%%
        
    # Callback that streams epoch results to a CSV file
    # https://keras.io/api/callbacks/csv_logger/
    csv_logger = CSVLogger(args['CSV_PATH'], append=True, separator=';')
    
    # serialize model to JSON
    try:
        model_json = model.to_json()
        with open(args['model_json'], 'w') as json_file:
            json_file.write(model_json)
    except:
        pass

    # Define whether the whole model or the weights only will be saved from the ModelCheckpoint
    # Refer to the documentation of ModelCheckpoint for extra details
    # https://keras.io/api/callbacks/model_checkpoint/
    
    temp = '{}_{}'.format(info, args['counter']) + "_epoch_{epoch}_" + \
            args['metric_to_plot'] + "_{val_" + args['metric_to_plot'] +":.3f}.h5"
    
    if args['save_model_weights'] == 'model':
        ModelCheckpoint_file = args["checkpoints"] + temp
        save_weights_only = False
    
    elif args['save_model_weights'] == 'weights':
        ModelCheckpoint_file = args['weights'] + temp
        save_weights_only = True
    
    epoch_checkpoint = EpochCheckpoint(args['checkpoints'], args['weights'], args['save_model_weights'],
                        every=args['every'], startAt=args['start_epoch'], info=info, counter=args['counter'])
    
    training_monitor = TrainingMonitor(args['FIG_PATH'], jsonPath=args['JSON_PATH'], 
                                       startAt=args['start_epoch'], metric=args['metric_to_plot'])
    
    model_checkpoint = ModelCheckpoint(ModelCheckpoint_file, monitor='val_{}'.format(args['metric_to_plot']), 
                                      verbose=1, save_best_only=True, mode='max', save_weights_only=save_weights_only)
        
    # Construct the set of callbacks
    callbacks = [csv_logger,
                 epoch_checkpoint,
                 training_monitor,
                 model_checkpoint]   
    
    #%%  
    # Train the network
    #
    
    H = model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BS,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // BS,
        epochs=epochs,
        max_queue_size=BS * 2,
        callbacks=callbacks, verbose=1)

#%%

elif mode == 'evaluate':

    # load pretrained model/weights
    from evaluate_class import LoadModel
    model = LoadModel(args, IMAGE_DIMS, BS).load_pretrained_model()

    # Do not use data augmentation when evaluating model: aug=None
    evalGen = HDF5DatasetGeneratorMask(args['EVAL_HDF5'], BS, aug=None, shuffle=False, binarize=args['binarize'])
    
    # Use the pretrained model to fenerate predictions for the input samples from a data generator
    predictions = model.predict_generator(evalGen.generator(),
                                          steps=evalGen.numImages // BS+1, max_queue_size=BS * 2, verbose=1)

    # Define folder where predictions will be stored
    predictions_folder = '{}{}/'.format(args['predictions'], args['pretrained_filename'])
    # Create folder where predictions will be stored
    cnf.check_folder_exists(predictions_folder)
    
    # Visualize  predictions
    # Create a plot with original image, ground truth and prediction
    # Show the metrics for the prediction
    # Output will be stored in a subfolder of the predictions folder (args['predictions_subfolder'])
    from subroutines.visualize_predictions import Visualize_Predictions
    Visualize_Predictions(args, predictions)

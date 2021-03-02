# Crack detection for masonry surfaces
This GitHub Repository was produced to share material relevant to the Journal paper **[Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning](https://doi.org/10.1016/j.autcon.2021.103606)** by **[D. Dais](https://www.researchgate.net/profile/Dimitris-Dais)**,  **İ. E. Bal**, **E. Smyrou**, and **V. Sarhosis** published in **Automation in Construction**.  

While in the paper models were trained both for patch classification and pixel segmentation, in the herein repository only the codes for crack segmentation are made available for the time being. Extra material regarding the patch classification will be added in the future.  

A sub-set of the masonry dataset used in the paper is made available as well in the **[dataset](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/dataset)** folder.  

Ιndicative examples of photos from our masonry dataset can be found in the folder **[dataset_sample](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/dataset_sample)**.  

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/Dais_et_al_Automatic_Crack_Detection_on_Masonry.png" width="700"> |
|:--:| 

The paper can be downloaded from the following links:
- [https://doi.org/10.1016/j.autcon.2021.103606](https://doi.org/10.1016/j.autcon.2021.103606)
- [https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning
](https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning)

In case you use or find interesting our work please cite the following Journal publication:

**D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. https://doi.org/10.1016/j.autcon.2021.103606.**

``` 
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
```  

Author and Moderator of the Repository: **[Dimitris Dais](https://github.com/dimitrisdais)**  

For further information please follow me in the below links  
LinkedIn: https://www.linkedin.com/in/dimitris-dais/  
Email: d.dais@pl.hanze.nl  
ResearchGate: https://www.researchgate.net/profile/Dimitris_Dais2  
Research Group Page: https://www.linkedin.com/company/earthquake-resistant-structures-promising-groningen  
YouTube Channel: https://www.youtube.com/channel/UCuSdAarhISVQzV2GhxaErsg  

Your feedback is welcome. Feel free to reach out to explore any options for collaboration.

# Table of Contents
 [Crack detection for masonry surfaces](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#crack-detection-for-masonry-surfaces)  
 [Table of Contents](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#table-of-contents)  
 [Publication - Brief Preview](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#publication---brief-preview)  
 | --- [Image patch classification for crack detection](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#image-patch-classification-for-crack-detection)  
 | --- [Crack segmentation on pixel level](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#crack-segmentation-on-pixel-level)  
 [How to download and run the code](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#how-to-download-and-run-the-code)  
 [Configuration file](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#configuration-file)  
 [Build data](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#build-data)  
 [Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)  
 [Evaluate](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#evaluate)  
 [Acknowledgements](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#acknowledgements)  
 [References](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#references)  

# Publication - Brief Preview
The aim of this study is to examine deep learning techniques for crack detection on images from masonry walls. A dataset with photos from masonry structures is produced containing complex backgrounds  and  various  crack  types  and  sizes.  Different  deep  learning  networks  are  considered  and  by leveraging the effect of transfer learning crack detection on masonry surfaces is performed on patch level with **95.3% accuracy** and on pixel level with **79.6% F1 score**. **This is the first implementation of deep learning for pixel-level crack segmentation on masonry surfaces.**

|<img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/crack_detection.png" width="700">|
|:--:| 

#### Image patch classification for crack detection
Different state of the art CNNs pretrained on ImageNet are examined herein for their efficacy to classify images from masonry surfaces on patch level as crack or non-crack.

|<img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/patch classification - crack.png" width="700">|
|:--:| 

|<img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/patch classification - non crack.png" width="700">|
|:--:| 

#### Crack segmentation on pixel level

[U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), a deep Fully Convolutional Network (FCN), and [Feature Pyramid Networks (FPN)](https://doi.org/10.1109/CVPR.2017.106), a generic pyramid representation, are considered herein and combined with different pretrained CNNs performing as the backbone of the encoder part of the network. Moreover, other networks found in the literature and performed well in crack segmentation are examined as well.

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/encoder-decoder architecture of Fully Convolutional Networks.png" width="700"> | 
|:--:| 
| *Schematic representation of the encoder-decoder architecture of Fully Convolutional Networks..* |

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/Feature  Pyramid  Network.png" width="700"> | 
|:--:| 
| *Schematic  representation  of  Feature  Pyramid  Network.* |

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/Unet.png" width="700"> | 
|:--:| 
| *Illustration of the architecture of U-net as implemented in the herein study.* |


| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/predictions.png" width="700"> | 
|:--:| 
|The original image, the ground truth and the prediction with U-net-MobileNet.| 

# How to download and run the code

Simply download the code and copy it your desired folder path.  
In order to use the code you need to define the **[Configuration file](https://github.com/dimitrisdais/crack_detection_CNN_masonry#configuration-file)** (**[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**) suitably.
There are three modes:  

- **[Build data](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#build-data)**  
- **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)**  
- **[Evaluate](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#evaluate)** 

To use the **[Build data](https://github.com/dimitrisdais/crack_detection_CNN_masonry#build-data)** mode you need to run the **[build_data.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/build_data.py)** file while the **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)** and **[Evaluate](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#evaluate)** modes are implemented with the **[train_evaluate.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/train_evaluate.py)** file. See the documentation below for extra info.  

# Configuration file

In the **[Configuration file](https://github.com/dimitrisdais/crack_detection_CNN_masonry#configuration-file)** (**[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**) you can define the parameters that will be used when running the code.  
See below some examples of different parameters that can be defined:  

``` python
        # Define the mode that will be used when running the code
        self.mode = 'evaluate' # 'train', 'evaluate' or 'build_data'
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
```  

When the **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)** mode is used, define which network to be used. The options are:  
- **[U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)**  
- **[Deeplabv3](https://github.com/tensorflow/models/tree/master/research/deeplab)**  
- **[DeepCrack](https://github.com/hanshenChen/crack-detection)**  
- **[Segmentation Models](https://github.com/qubvel/segmentation_models): Different models architectures for segmentation combined with pretrained CNNs performing as the backbone of the encoder part of the network**

``` python
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
```  

In order to use the **[Deeplabv3](
https://github.com/tensorflow/models/tree/master/research/deeplab)** network copy **[model.py](https://github.com/tensorflow/models/blob/main/research/deeplab/model.py)** to the **[networks](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/networks)** folder.  

In order to use the **[DeepCrack](https://github.com/hanshenChen/crack-detection)** network copy **[edeepcrack_cls.py](https://github.com/hanshenChen/crack-detection/blob/main/edeepcrack_cls.py)** and **[indices_pooling.py](https://github.com/hanshenChen/crack-detection/blob/main/indices_pooling.py)** to the **[networks](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/networks)** folder.

In order to use the **[Segmentation Models](https://github.com/qubvel/segmentation_models)** use the pip install method as shown in in the corresponding GitHub repository.  

|<img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/networks.png" width="350">|
|:--:|  
|Networks folder.|  

Is it possible to select among the following Loss Functions:
``` python
        # Define Loss Function
        # 'Focal_Loss'
        # 'WCE': Weighted Cross-Entropy
        # 'BCE': Binary Cross-Entropy
        # 'F1_score_Loss'
        # 'F1_score_Loss_dil': Background pixels predicted as cracks (FP) are considered as
        #                      TP if they are a few pixels apart from the annotated cracks
        #                      Refer to the Journal paper for extra clarification
        self.args['loss'] = 'WCE'
```  

# Build data
Under the folder **[dataset](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/dataset)** there are two sub-folders with the **[images](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/dataset/crack_detection_224_images)** and the corresponding **[masks](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main/dataset/crack_detection_224_masks)**.

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/images.png" width="700"> |
|:--:| 

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/masks.png" width="700"> |
|:--:| 

Execute the **[build_data.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/build_data.py)** to split the dataset into train and validation set and create HDF5 files that can be easily accessed when in **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)** and **[Evaluate](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#evaluate)** modes. The **output** folder will be created and the output files will be stored in the **hdf5** sub-folder. 

**It is noted that the output folder will be created automatically after executing the code.**  

Use the **[Build data](https://github.com/dimitrisdais/crack_detection_CNN_masonry#build-data)** mode by suitably setting the **[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**. 

``` python
        # Define the mode that will be used when running the code
        self.mode = 'build_data' # 'train', 'evaluate' or 'build_data'
``` 

In the **[build_data.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/build_data.py)** define appropriately the **[folder["main"]](https://github.com/dimitrisdais/crack_detection_CNN_masonry)** to correspond to the folder where you stored the repository on your local machine.

``` python
# The path where the repository is stored
folder['main'] = folder['initial'] + 'crack_detection_CNN_masonry/'

# if folder['main'] == '', then the current working directory will be used
if folder['main'] == '':
    folder['main'] = os.getcwd()
``` 

# Train
Use the **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)** mode by suitably setting the **[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**. 

``` python
        # Define the mode that will be used when running the code
        self.mode = 'train' # 'train', 'evaluate' or 'build_data'
``` 

In the **[train_evaluate.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/train_evaluate.py)** define appropriately the **[folder["main"]](https://github.com/dimitrisdais/crack_detection_CNN_masonry)** to correspond to the folder where you stored the repository on your local machine. Then execute the **[train_evaluate.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/train_evaluate.py)** to train the selected network on the provided dataset.

``` python
# The path where the repository is stored
folder['main'] = folder['initial'] + 'crack_detection_CNN_masonry/'

# if folder['main'] == '', then the current working directory will be used
if folder['main'] == '':
    folder['main'] = os.getcwd()
``` 

In order to be able to run different trials without overwriting any existing files, any output files of each trial will have a different suffix. This suffix will be obtained from the file named **counter.txt**. Create a txt file and type any number that will be used as a counter; during each trial you will be asked whether this value needs to be incremented by 1. If the **counter** file has not been created, then the os.getpid() will be used as counter.  

When training the model the following files will be created:

- The Loss/Metrics will be plotted during training to a png file in the folder **output**  
- The model will be serialized and stored in JSON format in the sub-folder **model_json**  
- The results (i.e. metrics, loss) will be serialized and stored in JSON format in the folder **output**  
- The architecture of the network will be stored in the folder **output**  
- The summary of the model will be stored to txt file in the folder **output**  
- The results (i.e. metrics, loss) will be stored to CSV format in the folder **output**  
- The weights of the trained model will be stored for different epochs in the sub-folder **weights**. See **[TrainingMonitor](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/subroutines/callbacks/trainingmonitor.py)** for extra details.  

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/output folder.png" width="350"> | 
|:--:| 
| *Output file created when **Train** mode is used.* |  

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/counter file.png" width="350"> | 
|:--:| 
| *The counter file.* |  

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/crack_detection_1.png" width="350"> | 
|:--:| 
| *The Loss/Metrics will be plotted during training.* |  

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/weights.png" width="350"> | 
|:--:| 
| *The weights of the trained model will be stored for different epochs.* |  

# Evaluate
Use the **[Evaluate](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#evaluate)** mode by suitably setting the **[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**. 

``` python
        # Define the mode that will be used when running the code
        self.mode = 'evaluate' # 'train', 'evaluate' or 'build_data'
``` 
  
In the **[train_evaluate.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/train_evaluate.py)** define appropriately the **[folder["main"]](https://github.com/dimitrisdais/crack_detection_CNN_masonry)** to correspond to the folder where you stored the repository on your local machine. 
If **[folder["main"]](https://github.com/dimitrisdais/crack_detection_CNN_masonry)** was set during the **[Train](https://github.com/dimitrisdais/crack_detection_CNN_masonry/tree/main#train)** mode, it does not need to be changed again.

``` python
# The path where the repository is stored
folder['main'] = folder['initial'] + 'crack_detection_CNN_masonry/'

# if folder['main'] == '', then the current working directory will be used
if folder['main'] == '':
    folder['main'] = os.getcwd()
```  

Define the file with the pretrained weights or the model with weights that will be used to evaluate the model by setting correctly **self.args['counter']** and **self.args['pretrained_filename']**.

``` python
            # Define the counter suitably in order to read the correct JSON file etc.
            self.args['counter'] = 1
            # Define the file with the pretrained weights or the model with weights that will be used to evaluate model
            # e.g. 'crack detection_1_epoch_7_F1_score_dil_0.762.h5'
            self.args['pretrained_filename'] = 'crack_detection_1_epoch_8_F1_score_dil_0.681.h5'
```   

The filename of the pretrained weights provides information regarding several parameters. For exampe from **'crack_detection_1_epoch_8_F1_score_dil_0.681.h5'** it can be inferred that:

self.info = **'crack_detection'**  
self.args['counter'] = **1**  
The epoch that the best value of the monitored metric was obtained: **8**  
Monitored metric: **F1_score_dil**  
Best value of the monitored metric: **0.681**  

For extra clarification, refer to the documentation of **[ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)**    

Subsequently, execute the **[train_evaluate.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/train_evaluate.py)** to use the pre-trained network to perform predictions and visualize them.

| <img src="https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/images/predictions_repo.png" width="350"> | 
|:--:| 
| *Predictions with the pretrained model.* |  

**It is noted that the metrics displayed in the figure above were obtained with training only for 10 epochs and a sub-set of the whole masonry dataset was used.**

When using the default configuration, the validation set will be used to evaluate the trained model. In case a different test needs to be used, then define suitably the **self.args['EVAL_HDF5']** in the **[config_class.py](https://github.com/dimitrisdais/crack_detection_CNN_masonry/blob/main/config_class.py)**.  

``` python
        # In case you need to test the model on a set other than the validation set,
        # define the EVAL_HDF5 suitably
        self.args['EVAL_HDF5'] = self.args['hdf5'] + temp + 'val.hdf5'
``` 

# Acknowledgements  

I would like to thank my friend and Deep Learning expert **[Nekartios Lianos](https://www.linkedin.com/in/nektarios-lianos-6a47148b/)** for his insightful guidance along my first steps in the world of CV and DL.  

I couldn not skip referring to the contribution of **[Adrian Rosebrock](https://www.linkedin.com/in/adrian-rosebrock-59b8732a/)**; the material he shares on his **[blog pyimagesearch](https://www.pyimagesearch.com/)** and his **[books](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)** ensure that the transition to the magical world of CV and DL will be as smooth as it can be.  

# References
The following codes are based on material provided by **[Adrian Rosebrock](linkedin.com/in/adrian-rosebrock-59b8732a)** shared on his blog (**https://www.pyimagesearch.com/**) and his books:

build_data.py  
hdf5datasetgenerator_mask.py  
hdf5datasetwriter_mask.py  
epochcheckpoint.py  
trainingmonitor.py  

- Adrian Rosebrock, Deep Learning for Computer Vision with Python - Practitioner Bundle, PyImageSearch, https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/, accessed on 24 February 2021  
- Adrian Rosebrock, Keras: Starting, stopping, and resuming training, PyImageSearch, https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/, accessed on 24 February 2021  
- Adrian Rosebrock, How to use Keras fit and fit_generator (a hands-on tutorial), PyImageSearch, https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/, accessed on 24 February 2021  

The Segmentation Models with pre-trained CNNs are implemented based on the work of **[Pavel Yakubovskiy](https://github.com/qubvel)** and his GitHub Repository https://github.com/qubvel/segmentation_models  

**DeepCrack** is implemented as provided by the corresponding [GitHub Repository](https://github.com/hanshenChen/crack-detection)  

**Deeplabv3** is implemented as provided by the corresponding [GitHub Repository](https://github.com/tensorflow/models/tree/master/research/deeplab)  

Unet is based on the implementation found in the link below:  
https://www.depends-on-the-definition.com/unet-keras-segmenting-images/  

The Weighted Cross-Entropy (WCE) Loss is based on the implementation found in the link below:  
https://jeune-research.tistory.com/entry/Loss-Functions-for-Image-Segmentation-Distribution-Based-Losses  

The Focal Loss is based on the implementation found in the link below:  
https://github.com/umbertogriffo/focal-loss-keras

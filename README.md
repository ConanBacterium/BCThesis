# Transfer Learning for Classification of Fungus on Microscopy Images using EfficientNet and Gradient Boosted Trees

## Abstract 
This paper examines the efficacy of an approach previously used successfully for the ICIAR 2018 histology dataset when
applied to a dataset provided by Teknologisk Institut of WSIs of mold from real world construction sites and private homes. The examined approach consists of extracting random crops (1300x1300 and 800x800) from the images, augmenting the brightness of the crops, extracting features from the crops with a pretuned EfficientNet B7 model, and pooling together the features into a single descriptor and training a Gradient Boosted Tree-model on the descriptors. The FungAI dataset is limited when considering the varied nature of the data and the varied nature of the appearance of mold, and it is a priori expected to be a harder problem than the ICIAR 2018 dataset, which the experiments of the current paper confirms. The approach does not have a good performance as measured by F1-score as compared to the ICIAR 2018, and experiments with increasing the train size doesn’t show a clear trend of improvement. Whether this is the approach is inherently poor suited for the FungAI problem, or the training size is much too low, or the quality of the annotations is too low, is not clear. Further research has to be conducted before definitive conclusions can be drawn.

## About the code
The dataset from Teknologisk Institut is private, and some of the code is private too. 

The general approach is available by running the preprocessing scripts iciar_preprocessing_aug.py and iciar_preprocessing_descriptors.py in sequence. They expect the following data directories: 

data
└── ICIAR2018_BACH_CHALLENGE
    └── Photos
        ├── Normal
        ├── Benign
        ├── InSitu
        └── Invasive

Then the cells of "Training GBT ICIAR.ipynb" can be executed. 

Note that it's possible to use get_descriptor_from_imgtensor__chunks_instead_of_random_crops() instead of get_descriptor_from_imgtensor() in the iciar_preprocessing_descriptors.py. The crops won't be extracted randomly, but instead the image will be cut into 4 equal parts, and then 9 equal parts, and those crops are then what are encoded by EfficientNet and pooled into a single descriptor. That way no information in the image will be lost. 


https://github.com/BMIRDS/deepslide 

* [19th of March 2023 (pre weekly-meetings](#march-19th)

* [12th of March 2023 (pre weekly-meetings](#march-12th)

* [7th March 2023 (progress as of 7th March, pre weekly-meetings)](#march-7th)

### March 19th

##### Main takeaways from meeting
* Find public dataset where this approach works and then compare with our data. (rad imagenet, grand-challenge patchcamelyon)
-- Downloading this rn: https://www.i3s.up.pt/digitalpathology/
---- big whole slide images as .svs files. Need to find out what zoom level looks most like FungAI images, to make sure they're as comparable as possible? 
---- need to figure that out with Ulf..... 

* Don't augment test set unless you're completely sure about what you're doing and can justify it. 

* Show that increasing train size increases accuracy. 

* Start writing report. 

##### Progress overall

1) Wrote ResNet50 from Scratch
2) Wrote custom pytorch datasets that can be balanced if needed
3) wrote training function that has early stopping with a learning rate scheduler
4) currently writing function to optimize threshold to given metric function
5) Wrote training script for finetuning EfficientNet, for finetuning ResNet50 by resizing training imgs to 224x224 and one for training ResNet50 from scratch. 

Need to do cross validation and let each fold become a model which will be put in ensemble. 

Need to write script to make masks from Paint.NET annotations. 

Need to write sliding window script. 

Need to write report. 

After that the bare minimum of the project is done and I can experiment to try and achieve a better model.

#### Questions: 

1) use ITU cluster to pretrain ResNet50 on ImageNet? Or wait for new annotations where 224x224 can be used and finetune the pretrained ResNet50 on that?
--- IRRELEVANT

2) If the overall approach (sliding window) is the same as the one in the project I'm replicating is it then okay to use a model that is different from theirs? Given that the EfficientNet seems to be better - which is to be expected, it's from 2019/2021 and ResNet is from 2015... 
---- YES IT'S OKAY

3) How to best insert metadata? 
--- DON'T INSERT METADATA, but add it as a label during training. This way the model learns what it is looking at. Adding it as input will lead to over fitting. 

4) Look at architecture to see if dimensions hold up, and maybe look at adding more layers to make the feature space smaller... Or maybe make the FC-layer bigger given that the feature space is big? 
--- DIDN'T DO THIS, probably not necessary. 

5) Can I unbalance the dataset, but balance the learning rate or loss function? So scale the lower-balanced class. Or maybe augment the lower-balanced class? hmm. 
--- AUGMENT lower-balanced class

6) Pytorch model weights are around 250MB for effnet and 100MB for ResNet50, and the batches of 20 pngs are around 8MB. I make them into tensors, so maybe they become bigger, so let's just say the batch is 50MB. This shouldn't be a lot of memory, yet if I increase batch size much more I get memory error... 
--- IRRELEVANT

7) Does batch normalization on many layers give weird results in a highly varied dataset with relatively small batches (20 imgs)... ?  
7.5) Normalizing the data makes many training examples unrecognizable, but it gives much higher accuracy? I guess this is fine, since no pattern in the data is removed, just rescaled.
--- IRRELEVANT

8) What to do with unsharp training examples? Skip or negative 
--- EXPERIMENT WITH IT... Try training as multilabel and try training without multilabel and compare, blabla. 

9) What does higher accuracy on unbalanced as compared to balanced mean? ...
--- IRRELEVANT

10) Is it a good idea to use balanced classes in development phase to reduce training time, and then when best parameters have been found you choose a distribution that looks like the real world?
--- SURE....? --- IRRELEVANT

Bonus question: is it somehow possible to pretrain specific filters on the 28x28 MEDMNIST dataset? Some of the images look kinda similar to our data. It would have to be the first convolutions I imagine. 
--- MAYBE, but don't worry about it

### March 12th

I have set up MLFlow. Had to fix VPN issues. 

Trying out segmentation annotations for full images with Paint.NET, hopefully it will speed up the annotation process. 

Tried balancing the datasets. Not enough annotations (~1500 per class), need more. And need to pretrain - we don't balance the finetuning training data for our current pretrained EffNet model! There is no pretrained ResNet for our image sizes (600x600). Our current model EffNet_b7 is pretrained and gets quite good results, ~84% accuracy. Currently downloading ImageNet to pretrain a ResNet, but the training will take too long on our system... 

UPDATE: 21:14 MAJOR BUGFIX! only returned annotation of index 19 in FungAIDataset __getitem__ method. 



### March 7th
As of march 7th there still hasn't been a meeting, as I have cancelled the meetings we've been supposed to have. 

Progress so far is that the annotation process has been sped up by orders of magnitude. In a few weeks we've gotten 3-5x more annotated data than we've collected the 12 previous months. That's very good. 

I've built a resnet50 model that works - that is to say that I can train it. The training examples are 600x600, whereas the architecture I'm using is built for 224x224, so my feature space is quite larger... So I will have to add more layers, and maybe make the filters bigger, change the learning rate, etc. 

I'm looking into setting up MLFlow as I would like to have an overview of my experiments, but maybe I will use another solution. 

Training for 1 and 2 epochs give the exact same result, 85% accuracy (which is about the same accuracy as the best working model we have now, using a finetuned EfficientNet). Training for 10 epochs gives accuracy of only 80%, so probably overfitting. I need to look more into that. 

Wei et. al. 2019 uses 224x224 size ResNet that is pretrained, but my constraint of 600x600 images means I have to change some stuff... They use batch size 64, but I can't do that as it doesn't fit into memory. Not sure if Tensorflow Records would help with that, as I imagine a whole batch will still need to be loaded at a time. Batch size 20 works fine on our system. 

Ibid. recommends a smaller ResNet18 model when there isn't much training data, but I don't see how that is necessary since the skip connections should mean the model won't degrade as the size increases... 
<hr>




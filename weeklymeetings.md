* [12th of March 2023 (pre weekly-meetings](#march-12th)

* [7th March 2023 (progress as of 7th March, pre weekly-meetings)](#march-7th)

### March 12th

I have set up MLFlow. Had to fix VPN issues. 

Trying out segmentation annotations for full images with Paint.NET, hopefully it will speed up the annotation process. 

Tried balancing the datasets. Not enough annotations (~1500 per class), need more. And need to pretrain - we don't balance the finetuning training data for our current pretrained EffNet model! There is no pretrained ResNet for our image sizes (600x600). Our current model EffNet_b7 is pretrained and gets quite good results, ~84% accuracy. Currently downloading ImageNet to pretrain a ResNet, but the training will take too long on our system... 

Questions: 

1) use ITU cluster to pretrain ResNet50 on ImageNet? Or wait for new annotations where 224x224 can be used and finetune the pretrained ResNet50 on that?

2) If the overall approach (sliding window) is the same as the one in the project I'm replicating is it then okay to use a model that is different from theirs? Given that the EfficientNet seems to be better - which is to be expected, it's from 2019/2021 and ResNet is from 2015... 

3) How to best insert metadata? 

4) It's a no-go when data augmentation produces positive training examples that an annotator would classify as negative, right?

Bonus question: is it somehow possible to pretrain specific filters on the 28x28 MEDMNIST dataset? Some of the images look kinda similar to our data. It would have to be the first convolutions I imagine. 



### March 7th
As of march 7th there still hasn't been a meeting, as I have cancelled the meetings we've been supposed to have. 

Progress so far is that the annotation process has been sped up by orders of magnitude. In a few weeks we've gotten 3-5x more annotated data than we've collected the 12 previous months. That's very good. 

I've built a resnet50 model that works - that is to say that I can train it. The training examples are 600x600, whereas the architecture I'm using is built for 224x224, so my feature space is quite larger... So I will have to add more layers, and maybe make the filters bigger, change the learning rate, etc. 

I'm looking into setting up MLFlow as I would like to have an overview of my experiments, but maybe I will use another solution. 

Training for 1 and 2 epochs give the exact same result, 85% accuracy (which is about the same accuracy as the best working model we have now, using a finetuned EfficientNet). Training for 10 epochs gives accuracy of only 80%, so probably overfitting. I need to look more into that. 

Wei et. al. 2019 uses 224x224 size ResNet that is pretrained, but my constraint of 600x600 images means I have to change some stuff... They use batch size 64, but I can't do that as it doesn't fit into memory. Not sure if Tensorflow Records would help with that, as I imagine a whole batch will still need to be loaded at a time. Batch size 20 works fine on our system. 

Ibid. recommends a smaller ResNet18 model when there isn't much training data, but I don't see how that is necessary since the skip connections should mean the model won't degrade as the size increases... 
<hr>




from torchvision import transforms
import torch 


resize_224_no_aug_no_norm = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

resize_224_with_aug_no_norm = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(224,padding=67),
    transforms.ToTensor(),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180)])

resize_600_no_aug_no_norm = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.ToTensor()])

resize_600_with_aug_no_norm = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(224,padding=67),
    transforms.ToTensor(),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180)])

resize_600_no_aug_with_norm = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

resize_600_with_aug_with_norm = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(224,padding=67),
    transforms.ToTensor(),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])






######### THIS IS DIRECTLY FROM OLD TRAIN SCRIPT
aug_preprocessing_3 = transforms.Compose([transforms.Resize(600), transforms.CenterCrop(600), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(600,padding=67), transforms.ToTensor(), transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), transforms.RandomRotation(180), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.RandomErasing()])

preprocess_effnetb7 = aug_preprocessing_3










######### OLD #########
fungai_preprocessing_w_augmentation = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(600,padding=67),
    transforms.ToTensor(),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    transforms.RandomErasing()])

fungai_preprocessing_without_augmentation = transforms.Compose([
    transforms.Resize(600), transforms.CenterCrop(600),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




fungai_preprocessing_resize_to_224_w_augmentation = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(224,padding=67),
    transforms.ToTensor(),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.3), 
    transforms.RandomRotation(180), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    transforms.RandomErasing()])

fungai_preprocessing_resize_to_224_without_augmentation = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

fungai_preprocessing_resize_to_224_without_augmentation_without_normalization = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

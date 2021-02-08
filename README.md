## Image Classification using VGG16 (Medical Imaging - Brain_MRI)

- Dataset - https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
- PretrainedModel - https://drive.google.com/file/d/1z6jhOK7gzT8J8mbIeCnV4tIg8xBMXw0I/view?usp=sharing

List of Tested Models With Accuracy (Augmentation_type => (flip))
Model - loss accuracy validation_loss validation_acc size_of_image
1. Resnet50 - 59% 66% 60% 63% (224 X 224)
2. Resnet50 - 65% 64% 65% 64% (256 X 256)
3. VGG16(10 epochs) - 46% 71% 44% 75% (224 X 224)             
4. VGG16(100 epochs with early stopping) - 30% 86% 27% 89% (224 X 224)              
 

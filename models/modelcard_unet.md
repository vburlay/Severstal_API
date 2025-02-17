# Model Card for Unet

Model for Image Segmentation with ResNet50

## Model date
Oct. 2024

## Model Details

### Model Description

- **Developed by:** V.Burlay (wladimir.burlay@gmail.com)
- **Model version:** 98 billion parameter model
- **Model type:** Unet with ResNet50 (Downsampling)
- **Finetuned from model [optional]:** Pretrained ResNet50 for image 
  classification 
  than fine-tuned with cross-entropy loss for binary classification

## Uses

### Direct Use

- Intended to be used for fun application, such as segmentation for defect 
  details
- Particulary intended for younger audience

## How to Get Started with the Model

model = load_model(model,custom_objects={
        'dice_coef':dice_coef})

## Training Details

Fine-tuning of ResNet50. Binary classification

### Training Data

For test used an evaluation data set of 12568 jpeg-files,
each with the size (1600,256). The files included both image without
defect and with defects of classes. Each image can have segment defects 
of class (ClassId = [1, 2, 3, 4]).

#### Preprocessing [optional]

It was used Preprocessing of Keras - The Size: 128,800

#### Training Hyperparameters

Optimizer - Adam
loss - 'binary_crossentropy',
metrics - [dice_coef])

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

For test used an evaluation data set of 5506 jpeg-files,
each with the size (1600,256). The files included both image without
defect and with defects of classes. Each image can have segment defects 
of class (ClassId = [1, 2, 3, 4]).

#### Factors
The model card lists the following factors as potentially
* Camera angle
* Presenter distance from camera
* Camera type
* Lighting

#### Metrics

The way to evaluate a predictions. The  *dice score*.
Recall that:
 $$Dice Score = 2 * \frac{area\_of\_overlap}{combined\_area}$$


## Technical Specifications [optional]

#### Hardware

GPU: Quadro P4000 with 7107 MB

#### Software

OS: Ubuntu 24.02, CUDA 12.06, Tensorflow 12.01

## Model Card Contact
e-mail: wladimir.burlay@gmail.com
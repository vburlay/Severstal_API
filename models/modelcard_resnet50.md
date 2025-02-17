# Model Card for ResNet50

Model for Image Classification (binary) ResNet50

## Model date
Oct. 2024

## Model Details

### Model Description

- **Developed by:** V.Burlay (wladimir.burlay@gmail.com)
- **Model version:** 26 billion parameter model
- **Model type:** Convolutional Neural Net
- **Finetuned from model [optional]:** Pretrained for image classification than fine-tuned with cross-entropy loss for binary classification

## Uses

### Direct Use

- Intended to be used for fun application, such as filter for defect details
- Particulary intended for younger audience

## How to Get Started with the Model

model = load_model(model)

## Training Details

Fine-tuning of ResNet50. Binary classification

### Training Data

For test used an evaluation data set of 12568 jpeg-files,
each with the size (1600,256). The files included both image without
defect and with defects of classes. Each image can have segment defects 
of class (ClassId = [1, 2, 3, 4]).

#### Preprocessing [optional]

It was used Preprocessing of Keras - The Size 224,224

#### Training Hyperparameters

Optimizer - RMSprop(learning_rate=1e-3)
loss - 'binary_crossentropy',
metrics - ['accuracy'])

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

Main metrics(for one batch):

    precision    recall  f1-score   support

         0.0       0.97      0.94      0.95        31
         1.0       0.94      0.97      0.96        33

    accuracy                           0.95        64
    macro avg       0.95      0.95     0.95        64
    weighted avg    0.95      0.95     0.95        64

AUC: 0.9525904203323559

## Technical Specifications [optional]

#### Hardware

GPU: Quadro P4000 with 7107 MB

#### Software

OS: Ubuntu 24.02, CUDA 12.06, Tensorflow 12.01

## Model Card Contact
e-mail: wladimir.burlay@gmail.com
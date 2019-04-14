# Dog Breed Classifier
This project is part of Udacity's Data Scientist Nanodegree. The problem which will be solved in this project is to identify the breed of dogs using convolutional neural network. The goal is that builds a pipeline that takes an image as input and detects whether or not the image contains human or dog, then predicts the breed of the dog or the human resembles dog breed.

# Overview
![pipeline](images/pipline_overview.png)
The pipeline includes three models: 
- Dog detector: A binary classifier. Uses a ResNet-50 pre-trained on ImageNet. Since the model output 1000 classes, as long as the output falls into one of the dog categories, the model will output `TRUE`
- Human face detector: A binary classifier. Uses a pre-trained `FACENET` to detect the human face.

The image will be passed into `Dog Breed Classifier` when either `Dog detector` or `Human face dectector` output `True`, otherwise will request input a new image to start over. 

## Approach
The project use `Transfer Learning` leverage the imagenet pre-trained weights. The Keras-Application provides several pre-trained networks such as: VGG-19, ResNet-50, InceptionV3 and Xception. I prefer `Xception` because it introduces group convolution and residual connection as improvements for InceptionV3, and VGG-19 is very slow compare to others.

```python
backbone = xception.Xception(weights='imagenet', include_top=False)
classifier = Sequential()
classifier.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
classifier.add(Dense(133, activation='softmax'))
classifier.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 2048)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 133)               272517    
=================================================================
Total params: 272,517
Trainable params: 272,517
Non-trainable params: 0
_________________________________________________________________
```

Xception is actually a backbone in the `Dog Breed Classifier` for image feature extractor, for example:
```python
# extract image features through backbone
img_features = backbone.predict(xception.preprocess_input(img))
# obtain predicted vector
pred = classifier.predict(img_features)
```

### metrics
```Accuracy = (TP + TN) / (TP + TN + FP + FN)```

### training
Since only re-train the last layer, I used a relative small learning rate (`5e-3`) and trained with SGD optimizer for 400 epoches. The training finally achieved: 
```bash
# training
loss: 0.1026
accuracy: 0.9904
# validation
val loss: 0.4330
val accuracy: 0.8587
```

## Requirements
- Python 3.6
- Keras + Tensorflow
- Flask

### installation
```bash
pip install -r requirements.txt
```

## Web application
A web application been provided for running the inference on the dog breed classification.

### Through docker

- Build docker image
```
docker build -t dog-breed .
```
- Run the container
```
docker run -it -p 3001:3001 dog-breed
```
### Directly run with Python
```bash
cd webapp
python3 run_server.py
```
Access the web app through the URL: `http://<ip>:3001`

## Results
With Xception pre-trained model, achieved `85.77%` accuracy on the test set. 

(Have tried ResNet-50 pre-trained model which reached around `80%` accuracy)

## Discussion
Appearently, there are several ways can push the accuracy further more:
- Improve the accuracy of dog breed > 90% by either use data augmentation or more sophisticated pre-trained model like ResNeXt
- Use Ensemble approach that average the outputs from multiple individual DNN to increase the accuracy.
- The current pipeline uses three DNNs (human detector, dog detector and dog breed classifier) which result in slow inference speed. If possible, train an end-to-end network would be better for inference speed.

## Screenshots
![Web app homepage](images/web_screenshot.png)
![Result-1](images/result1.png)
![Result-2](images/result2.png)
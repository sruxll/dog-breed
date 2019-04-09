# Dog Breed Classification

A image classifier for identifying dog breed. Given an image of a dog, the classifier will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.

# Overview
![Web app homepage](images/web_screenshot.png)


## Requirements
- Python 3.6
- Keras + Tensorflow
- Flask

### installation
```bash
pip install -r requirements.txt
```

## Run Web app
A web app been provided for running the inference on the dog breed classification.

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
![Result-1](images/result1.png)
![Result-2](images/result2.png)
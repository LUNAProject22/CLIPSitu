# Clip Embeddings for Role Prediction in the IMSitu Dataset
This project contains code for training and evaluating several models for predicting roles based on Clip embeddings of images and verbs, using the IMSitu dataset.

## Dataset
The IMSitu dataset contains images with corresponding verbs and roles. Each verb can have up to six roles associated with it. The goal is to predict the roles given the image and the verb.

We generate Clip embeddings for each image and verb, and also for each role. For each image-verb pair, we concatenate the image embedding, verb embedding, and up to six role embeddings (if they exist) to obtain a sequence representation. We then feed this sequence through various models to predict the roles.

## Models
We implement three different models for role prediction:

GGNN (Graph-based Global Reasoning Neural Network): A graph neural network that uses the IMSitu graph structure to perform global reasoning over the roles.

Transformer: A transformer-based model that uses self-attention to process the variable-length input sequence.

MLP (Multi-Layer Perceptron): A simple feed-forward neural network that takes a fixed-length concatenation of the image, verb, and role embeddings as input.

## Training and Evaluation
We train each model on the training split of the IMSitu dataset, and evaluate it on the validation split. We report accuracy and F1 score for role prediction.

We provide scripts for training and evaluating each model, along with hyperparameter configurations.

Requirements
- Python 3.6 or higher
- PyTorch 1.8 or higher
- Transformers library
- NumPy
- Pandas
- tqdm

## Usage

To train a model, run:
```
python train.py --model <model_name> --epochs <num_epochs> --batch_size <batch_size>
```
To evaluate a model, run:
```
python evaluate.py --model <model_name>
```
Replace <model_name> with ggnn, transformer, or mlp depending on which model you want to use. You can also specify additional hyperparameters like learning rate, weight decay, and hidden size.
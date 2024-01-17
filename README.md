# ClipSitu: Effectively Leveraging CLIP for Conditional Predictions in Situation Recognition
This project contains code for training and evaluating several models for predicting roles based on Clip embeddings of images and verbs, using the IMSitu dataset.

## Dataset
We perform our experiments on imSitu dataset and the augmented imSitu dataset called SWiG for situation recognition and localization, respectively termed as grounded situation recognition. The dataset has a total of 125k images with 75k train, 25k validation, and 25k test images. The metrics used for semantic role labeling are value and value-all which predict the accuracy of noun. 

## Models
We implement three different models.

ClipSitu XTF: Cross-Attention Transformer

ClipSitu TF: ClipSitu Transformer

ClipSitu MLP

## Code

Please download the code here. https://drive.google.com/drive/folders/1mUqBRu6-ncGz65LHAaEeGP6tQox-tyGI?usp=sharing

## Training and Evaluation
We train each model on the training split of the IMSitu dataset and evaluate it on the validation split. 

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

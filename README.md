Visual Transformer on FashionMNIST

This repository contains an implementation of a Vision Transformer (ViT) model trained and evaluated on the FashionMNIST dataset.

Files

train.py - Trains the Vision Transformer model with specified hyperparameters and saves the trained weights to model.pth.

eval.py - Loads the saved model.pth weights and performs inference on test data, reporting classification performance.

Usage

Training

python train.py

This will train the model on the FashionMNIST training set and save the learned parameters to model.pth.

Evaluation

python eval.py

This will load model.pth and run inference on the FashionMNIST test set, outputting accuracy and other metrics.

Requirements

Python 3.7+

PyTorch

torchvision

Install dependencies with:

pip install -r requirements.txt

License

This project is released under the MIT License. Feel free to use and modify.

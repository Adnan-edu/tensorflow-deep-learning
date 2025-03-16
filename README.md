# TensorFlow Deep Learning ![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg)

A comprehensive guide to deep learning with TensorFlow, covering essential concepts from fundamentals to advanced topics including computer vision, natural language processing, and time series forecasting.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Course Content](#course-content)
  - [00: TensorFlow Fundamentals](#00-tensorflow-fundamentals)
  - [01: Neural Network Regression](#01-neural-network-regression)
  - [02: Neural Network Classification](#02-neural-network-classification)
  - [03: Computer Vision with TensorFlow](#03-computer-vision-with-tensorflow)
  - [04: Transfer Learning Part 1: Feature Extraction](#04-transfer-learning-part-1-feature-extraction)
  - [05: Transfer Learning Part 2: Fine-Tuning](#05-transfer-learning-part-2-fine-tuning)
  - [06: Transfer Learning Part 3: Scaling Up](#06-transfer-learning-part-3-scaling-up)
  - [07: Milestone Project 1: Food Vision Big™](#07-milestone-project-1-food-vision-big)
  - [08: Natural Language Processing with TensorFlow](#08-natural-language-processing-with-tensorflow)
  - [09: Milestone Project 2: SkimLit](#09-milestone-project-2-skimlit)
  - [10: Milestone Project 3: BitPredict](#10-milestone-project-3-bitpredict)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Exercises and Solutions](#exercises-and-solutions)
- [Resources](#resources)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Introduction

This repository contains comprehensive materials for learning deep learning with TensorFlow. The content progresses from fundamental concepts to advanced applications across various domains including computer vision, natural language processing, and time series forecasting.

The course is designed to provide both theoretical understanding and practical implementation skills through hands-on notebooks, exercises, and milestone projects that tackle real-world problems.

## Repository Structure

```
tensorflow-deep-learning/
├── 00_tensorflow_fundamentals.ipynb
├── 00_tensorflow_fundamentals_exercises_solutions.ipynb
├── 01_neural_network_regression_in_tensorflow.ipynb
├── 01_neural_network_regression_in_tensorflow_exercises_solutions.ipynb
├── 02_neural_network_classification_in_tensorflow.ipynb
├── 02_neural_network_classification_in_tensorflow_exercises_solutions.ipynb
├── 03_introduction_to_computer_vision_with_tensorflow.ipynb
├── 04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb
├── 04_transfer_learning_in_tensorflow_part_1_feature_extraction_exercises_solutions.ipynb
├── 05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb
├── 05_transfer_learning_in_tensorflow_part_2_fine_tuning_exercises_solutions.ipynb
├── 06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb
├── 07_food_vision_milestone_project_1.ipynb
├── 08_natural_language_processing_with_tensorflow.ipynb
├── 08_natural_language_processing_with_tensorflow_exercises.ipynb
├── 09_skimlit_nlp_milestone_project_2.ipynb
├── 10_time_series_forecasting_in_tensorflow.ipynb
├── 10_time_series_forecasting_in_tensorflow_exercise_solutions.ipynb
```

## Getting Started

To begin exploring this repository:

1. Clone the repository
2. Set up your environment with the required dependencies
3. Open the notebooks in sequential order, starting with `00_tensorflow_fundamentals.ipynb`
4. Follow along with the code examples and complete the exercises

## Course Content

### 00: TensorFlow Fundamentals

**File:** `00_tensorflow_fundamentals.ipynb`

This section introduces the core concepts of TensorFlow and tensor operations.

**Topics covered:**

- Introduction to tensors and their creation
- Tensor attributes and information extraction
- Tensor manipulation operations
- Interoperability between TensorFlow and NumPy
- Using `@tf.function` decorator to accelerate Python functions

### 01: Neural Network Regression

**File:** `01_neural_network_regression_in_tensorflow.ipynb`

This section focuses on building neural networks for regression problems using TensorFlow.

**Topics covered:**

- Understanding regression problems
- Architecture of regression models
- Input/output shapes
- Data creation and preparation
- Model creation, compilation, and fitting
- Loss functions and optimizers
- Model evaluation and visualization
- Model saving and loading

If you were working on building a machine learning algorithm for predicting housing prices, your inputs may be number of bedrooms, number of bathrooms and number of garages, giving you an input shape of 3 
<img src="https://camo.githubusercontent.com/248aeb7b7526acb77e5832ee3b9cfd7acf6b5a7d67c23133718b601b2a9ece3e/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30312d696e7075742d616e642d6f75747075742d7368617065732d686f7573696e672d7072696365732e706e67"
width="800">
### 02: Neural Network Classification

**File:** `02_neural_network_classification_in_tensorflow.ipynb`

This section explores classification problems and how to solve them with neural networks.

**Topics covered:**

- Binary and multi-class classification
- Architecture of classification models
- Loss functions for classification
- Evaluation metrics for classification models
- The power of non-linearity in neural networks
- Visualization techniques for classification problems

Example workflow of how a supervised neural network starts with random weights and updates them to better represent the data by looking at examples of ideal outputs.

<img src="https://camo.githubusercontent.com/25f5febe3d29db0c31c94487108302b109aafe8695fd5664ea369c3ef238abea/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30322d66617368696f6e2d6d6e6973742d6c6561726e696e672e706e67"
width="800">

### 03: Computer Vision with TensorFlow

**File:** `03_introduction_to_computer_vision_with_tensorflow.ipynb`

This section introduces convolutional neural networks (CNNs) for computer vision tasks.

**Topics covered:**

- Architecture of convolutional neural networks
- Binary image classification with CNNs
- Multi-class image classification with CNNs
- Data preparation for image tasks
- Building baseline CNN models
- Evaluating and improving computer vision models

A simple example of how you might stack together the above layers into a convolutional neural network. Note the convolutional and pooling layers can often be arranged and rearranged into many different formations.
<img src="https://camo.githubusercontent.com/b14f3faa144f22dfe7765324f43b063bbae9fdba485ccadee19d95411762239a/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30332d73696d706c652d636f6e766e65742e706e67"
width="800">


### 04: Transfer Learning Part 1: Feature Extraction

**File:** `04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb`

This section introduces transfer learning as a technique to leverage pre-trained models.

**Topics covered:**

- Introduction to transfer learning concepts
- Feature extraction approach
- Using TensorFlow Hub for pre-trained models
- Working with smaller datasets for rapid experimentation
- TensorBoard integration for tracking model performance

What we're working towards building. Taking a pre-trained model and adding our own custom layers on top, extracting all of the underlying patterns learned on another dataset our own images.

<img src="https://camo.githubusercontent.com/15b4cdcc71adc71306b7e5de999e21a0611b65608ba498c43db29b3285a1425c/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30342d7472616e736665722d6c6561726e696e672d666561747572652d65787472616374696f6e2e706e67"
width="800">

The different kinds of transfer learning. An original model, a feature extraction model (only top 2-3 layers change) and a fine-tuning model (many or all of original model get changed).

<img src="https://camo.githubusercontent.com/e2a1f0a8929da764eb233e93be3447eb636e3229ee40be3ea18581698b25518d/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30342d646966666572656e742d6b696e64732d6f662d7472616e736665722d6c6561726e696e672e706e67"
width="800">

A ResNet50V2 backbone with a custom dense layer on top (10 classes instead of 1000 ImageNet classes). Note: The Image shows ResNet34 instead of ResNet50. Image source: https://arxiv.org/abs/1512.03385

<img src="https://camo.githubusercontent.com/7af9b1df8e578b3782b3d439363210837ccd55cba80af393b982eaadd710dfb6/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30342d7265736e65742d666561747572652d657874726163746f722e706e67"
width="800">

### 05: Transfer Learning Part 2: Fine-Tuning

**File:** `05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb`

This section expands on transfer learning by exploring fine-tuning techniques.

**Topics covered:**

- Feature extraction vs. fine-tuning
- Keras Functional API for model building
- Data augmentation techniques
- Model experimentation methodology
- ModelCheckpoint callback for saving training progress
- Comparative analysis of different approaches

Feature extraction transfer learning vs. fine-tuning transfer learning. The main difference between the two is that in fine-tuning, more layers of the pre-trained model get unfrozen and tuned on custom data. This fine-tuning usually takes more data than feature extraction to be effective.

<img src="https://camo.githubusercontent.com/ecfd3428b2b1bb2798605077d017cc34e815c4cbed0c5b0d999966681becff91/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30352d7472616e736665722d6c6561726e696e672d666561747572652d65787472616374696f6e2d76732d66696e652d74756e696e672e706e67"
width="800">

High-level example of fine-tuning an EfficientNet model. Bottom layers (layers closer to the input data) stay frozen where as top layers (layers closer to the output data) are updated during training.

<img src="https://camo.githubusercontent.com/a195d5e5a1dd9fd30317eb579ed435d59af3e61d5b838d0dfb9ed6abe0304aab/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30352d66696e652d74756e696e672d616e2d656666696369656e746e65742d6d6f64656c2e706e67"
width="800">

### 06: Transfer Learning Part 3: Scaling Up

**File:** `06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb`

This section demonstrates how to scale transfer learning models to larger datasets.

Machine learning practitioners are serial experimenters. Start small, get a model working, see if your experiments work then gradually scale them up to where you want to go (we're going to be looking at scaling up throughout this notebook).

<img src="https://camo.githubusercontent.com/8ae41cd33b6beff300dfd42c464ce7d2b65fd4aa7747023c8e151571a139acc5/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30362d6d6c2d73657269616c2d6578706572696d656e746174696f6e2e706e67"
width="800">

**Topics covered:**

- Working with larger datasets
- Training feature extraction models on scale
- Fine-tuning at scale
- Model saving and loading strategies
- Performance evaluation
- Error analysis and model improvement

<a id="07-milestone-project-1-food-vision-big"></a>
### 07: Milestone Project 1 🍔👁 : Food Vision Big™

**File:** `07_food_vision_milestone_project_1.ipynb`

This milestone project applies the learned concepts to build a food classification system.

**Topics covered:**

- Using TensorFlow Datasets
- Preprocessing and data pipelining
- Performance optimization techniques
- Mixed precision training
- Building and fine-tuning large-scale models
- Beating benchmark performance

**Project goal:** Build a model to classify food images across 101 classes, aiming to beat the DeepFood paper's 77.4% top-1 accuracy.

### 08: Natural Language Processing with TensorFlow

A handful of example natural language processing (NLP) and natural language understanding (NLU) problems. These are also often referred to as sequence problems (going from one sequence to another).

<img src="https://camo.githubusercontent.com/b8da81f15f1f6616c5494867c3a8d69f1c0c4bff9632fd5a8e16d86cd17d486f/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30382d6578616d706c652d6e6c702d70726f626c656d732e706e67"
width="800">

**File:** `08_natural_language_processing_with_tensorflow.ipynb`

This section introduces fundamental NLP concepts and techniques using TensorFlow.

**Topics covered:**

- Text preprocessing and tokenization
- Word embeddings
- Sequential data modeling
- Universal Sentence Encoder
- Combining models through ensembling
- Confusion matrix analysis for text classification

Example text classification inputs and outputs for the problem of classifying whether a Tweet is about a disaster or not.

<img src="https://camo.githubusercontent.com/8e5975362dc9b16e4c326bdecd31312bfd237f7529372034188a6789fb5fbd55/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30382d746578742d636c617373696669636174696f6e2d696e707574732d616e642d6f7574707574732e706e67"
width="800">

Coloured block example of the structure of an recurrent neural network.

<img src="https://camo.githubusercontent.com/b7f28f4799fababd975053019ed8cfcb8b6980c685035edd5fd3c7d2f17c3209/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30382d524e4e2d6172636869746563747572652d636f6c6f757265642d626c6f636b2d65646974696f6e2e706e67"
width="800">

<a id="09-milestone-project-2-skimlit"></a>
### 09: Milestone Project 2: SkimLit 📄🔥

**File:** `09_skimlit_nlp_milestone_project_2.ipynb`

This milestone project focuses on sequential sentence classification in medical abstracts.
Example inputs (harder to read abstract from PubMed) and outputs (easier to read abstract) of the model we're going to build. The model will take an abstract wall of text and predict the section label each sentence should have.

<img src="https://camo.githubusercontent.com/6be872ade0fa4df8cd01323399c130e80cb32d6f179ed4d46b1f8f79c7c1ac42/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f30392d736b696d6c69742d6f766572766965772d696e7075742d616e642d6f75747075742e706e67"
width="800">

**Topics covered:**

- Sequential sentence classification
- Multimodal model architecture
- Token embeddings, character embeddings, positional embeddings
- Building baseline models with TF-IDF
- Error analysis and improvement

**Project goal:** Replicate the deep learning model from the paper "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts" to improve readability of medical research abstracts.

<a id="10-milestone-project-3-bitpredict"></a>
### 10: Milestone Project 3: BitPredict 💰📈

**File:** `10_time_series_forecasting_in_tensorflow.ipynb`

Time series problems deal with data over time.

Such as, the number of staff members in a company over 10-years, sales of computers for the past 5-years, electricity usage for the past 50-years.

The timeline can be short (seconds/minutes) or long (years/decades). 

<img src="https://camo.githubusercontent.com/aa793294e35f8601103020138060fe258c490b31645e1771b1c37a7759c023a0/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f31302d6578616d706c652d74696d652d7365726965732d70726f626c656d732e706e67"
width="800">

| Problem Type   | Examples                                                               | Output              |
|---------------|------------------------------------------------------------------------|---------------------|
| Classification | Anomaly detection, time series identification (where did this time series come from?) | Discrete (a label) |
| Forecasting    | Predicting stock market prices, forecasting future demand for a product, stocking inventory requirements | Continuous (a number) |

<img src="https://camo.githubusercontent.com/efa1b59754b0459d4e3e2c27b321be4bed12c4da6b9a2b6beb3a65ef7eaea02d/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f31302d74797065732d6f662d74696d652d7365726965732d7061747465726e732e706e67"
width="800">

**This milestone project explores time series forecasting for cryptocurrency price prediction.**

N-BEATS algorithm:

One of the best ways to improve a model's performance is to increase the number of layers in it.

That's exactly what the [N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) algorithm](https://arxiv.org/pdf/1905.10437.pdf) does.

The N-BEATS algorithm focuses on univariate time series problems and achieved state-of-the-art performance in the winner of the M4 competition (a forecasting competition).

For our next modelling experiment we're going to be replicating the generic architecture of the N-BEATS algorithm (see section [3.3 of the N-BEATS paper](https://arxiv.org/pdf/1905.10437.pdf)).

We're not going to go through all of the details in the paper, instead we're going to focus on:

 1. Replicating the model architecture in Figure 1 of the N-BEATS paper
    <img src="https://camo.githubusercontent.com/e99bc0fa7c0be2c3d8e9107abedb9dde84564ca0f4ca55f835b2a2eebce3d3d7/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d7264626f75726b652f74656e736f72666c6f772d646565702d6c6561726e696e672f6d61696e2f696d616765732f31302d6669677572652d312d6e62656174732d70617065722d616e6e6f74617465642e706e67"
width="800">

    N-BEATS algorithm we're going to replicate with TensorFlow with window (input) and horizon (output) annotations.
    
2. Using the same hyperparameters as the paper which can be found in [Appendix D of the N-BEATS paper](https://arxiv.org/pdf/1905.10437.pdf)

**Topics covered:**

- Time series fundamentals
- Data preparation for time series problems
- Train/test splitting for temporal data
- Windowing techniques
- Dense, LSTM, and 1D CNN models for time series
- Multivariate time series analysis
- Ensemble methods for forecasting
- Prediction intervals and uncertainty estimation
- The N-BEATS algorithm implementation

**Project goal:** Build models to forecast Bitcoin prices, while understanding the limitations of such predictions in open systems.

## Prerequisites

- Python 3.8+
- Understanding of machine learning concepts
- Basic knowledge of neural networks
- Familiarity with Python programming

## Installation

```bash
# Clone the repository
git clone https://github.com/Adnan-edu/tensorflow-deep-learning
cd tensorflow-deep-learning

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Usage

The notebooks are designed to be self-contained and can be run in sequence. Each notebook includes detailed explanations, code examples, and visualizations.

```bash
# Start Jupyter Notebook
jupyter notebook

# Or JupyterLab
jupyter lab
```

<a id="exercises-and-solutions"></a>
## ![Exercises Badge](https://img.shields.io/badge/Exercises-Solutions-blue?style=flat-square)

Each main topic includes exercise notebooks with solutions to help reinforce learning. These exercises challenge you to apply the concepts covered in the main notebooks to similar but slightly different problems.

**Exercise Files:**

- `00_tensorflow_fundamentals_exercises_solutions.ipynb`
- `01_neural_network_regression_in_tensorflow_exercises_solutions.ipynb`
- `02_neural_network_classification_in_tensorflow_exercises_solutions.ipynb`
- `04_transfer_learning_in_tensorflow_part_1_feature_extraction_exercises_solutions.ipynb`
- `05_transfer_learning_in_tensorflow_part_2_fine_tuning_exercises_solutions.ipynb`
- `08_natural_language_processing_with_tensorflow_exercises.ipynb`
- `10_time_series_forecasting_in_tensorflow_exercise_solutions.ipynb`

<a id="resources"></a>
## ![Resources Badge](https://img.shields.io/badge/Resources-Papers-blue?style=flat-square)

### Additional Reading

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### Research Papers Referenced

1. [Food101 Paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)
2. [DeepFood Paper](https://arxiv.org/abs/1606.05675)
3. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)
4. [Neural basis expansion analysis for interpretable time series forecasting (N-BEATS)](https://arxiv.org/abs/1905.10437)

<a id="credits"></a>
## ![Credits Badge](https://img.shields.io/badge/Credits-DanielBourke-blue?style=flat-square)

The content is based on Daniel's comprehensive deep learning course and reflects his expertise in making complex deep learning concepts accessible through practical, hands-on examples.

Visit [Daniel's GitHub profile](https://github.com/mrdbourke) for more resources on machine learning and deep learning.


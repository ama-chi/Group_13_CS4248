# Group 13 CS4248 Machine Comprehension Model

Welcome to the Group 13 CS4248 repository! This project is a collaborative effort by a team of five dedicated individuals working on a machine comprehension model. Machine reading comprehension model takes in a Question and an associated Passage, and outputs a segment in the passage (A word, sentence or paragraph) containing the answer to the question.
Our goal is to explore different ways we can build a lighter weight Machine Reading Comprehension model, by decreasing training time, training data required or model size while enhancing accuracy.

## Overview

In this repository, we employ a popular approach in Natural Language Processing (NLP) by leveraging frozen pre-trained models, such as BERT, ALBERT, and DistilBERT. We enhance these models by adding a final layer to fine-tune and optimize their performance for the specific task of machine comprehension. The final layer helps our models better understand the context and generate more accurate answers to questions posed to them.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)


## Setup

To get started with our machine comprehension model, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YourUsername/Group_13_CS4248.git
   cd Group_13_CS4248
   ```

2. **Environment Setup:**

   Set up a Python environment and install the required packages. You can create a virtual environment to manage dependencies. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models:**

   Download the pre-trained BERT, ALBERT, and DistilBERT models you intend to use. You can find these models on the Hugging Face Transformers library or other sources. Make sure to update the model paths in your code accordingly.

4. **Run the Model:**

   Execute your machine comprehension model by running the main script or Jupyter notebook files. Make sure to provide the necessary input data, including the text and questions for testing.

## Usage

In this section, provide detailed instructions on how to use your machine comprehension model. Include code examples and explanations on how to:

- Load pre-trained models.
- Preprocess input data.
- Fine-tune the models with your final layer.
- Ask questions and retrieve answers.
- Evaluate the model's performance.

You may also want to include some sample code snippets to demonstrate how to use your model effectively.



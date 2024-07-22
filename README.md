# ReviewSentimentAnalyzer

This project focuses on classifying restaurant reviews as positive or negative using a Random Forest classifier. The project includes text preprocessing, creating a Bag of Words model, training the classifier, and optimizing the model using Grid Search.

## Project Overview

The project uses a dataset of restaurant reviews, processes the text to remove noise, creates a Bag of Words model, and trains a Random Forest classifier to predict the sentiment of reviews. The classifier is then optimized using Grid Search to find the best hyperparameters.

## Project Structure

- **Importing the libraries**
- **Importing the dataset**
- **Cleaning the texts**
- **Creating the Bag of Words model**
- **Splitting the dataset into the Training set and Test set**
- **Training the Random Forest model on the Training set**
- **Applying Grid Search to find the best model and the best parameters**
- **Predicting the Test set results**
- **Making the Confusion Matrix**
- **Predicting a single Review**

## Results

The model achieved an accuracy of **75.2%** on the test set. The best accuracy achieved using Grid Search is **79.33%**.

## Requirements

- numpy
- matplotlib
- pandas
- nltk
- scikit-learn

You can install these packages using pip:

```bash
pip install numpy matplotlib pandas nltk scikit-learn
```
## Dataset
The dataset used is Restaurant_Reviews.tsv, which contains restaurant reviews and their corresponding sentiment. It has two columns:

Review: The text of the restaurant review.
Liked: A binary value indicating whether the review is positive (1) or negative (0).

## How to Run
Clone the repository.
Install the required libraries.
Place the dataset Restaurant_Reviews.tsv in the project directory.
Run the script to train the model and predict the sentiment of a review.
```bash
python ReviewSentimentAnalyzer.py
```


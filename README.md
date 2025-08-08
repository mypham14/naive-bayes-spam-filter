# SMS Spam Filter using Naive Bayes

## Project Overview

This project involves building a spam filter for SMS messages using the multinomial Naive Bayes algorithm. The goal is to classify new messages with an accuracy greater than 80%, distinguishing between spam and ham (non-spam) messages.

## Dataset

- **Source:** The dataset consists of 5,572 SMS messages classified by humans, compiled by Tiago A. Almeida and José María Gómez Hidalgo. It is available from the [UCI Machine Learning Repository](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/#composition).
- **Distribution:** Approximately 87% of the messages are ham, and 13% are spam.

## Project Steps

### 1. Data Exploration
- Load and explore the dataset to understand its structure and distribution.
- The dataset contains two columns: `Label` (spam or ham) and `SMS` (the message content).

### 2. Training and Test Set Creation
- Randomize and split the dataset into training (80%) and test sets (20%).
- Verify the distribution of spam and ham messages in both sets.

### 3. Data Cleaning
- Remove punctuation and convert all text to lowercase.
- Tokenize the SMS messages into individual words.

### 4. Vocabulary Creation
- Create a vocabulary of unique words from the training set.

### 5. Feature Engineering
- Transform the training set into a format where each row represents a message and each column represents a word from the vocabulary.

### 6. Calculating Probabilities
- Calculate the prior probabilities $$ P(\text{Spam}) $$ and $$ P(\text{Ham}) $$.
- Use Laplace smoothing to calculate the likelihoods $$ P(w_i|\text{Spam}) $$ and $$ P(w_i|\text{Ham}) $$.

### 7. Classifying New Messages
- Implement a function to classify new messages based on the calculated probabilities.
- Test the classifier on the test set and achieve an accuracy of 98.74%.

## Next Steps

- Analyze the 14 messages that were incorrectly classified to understand why the algorithm made mistakes.
- Enhance the model by making it sensitive to letter case and exploring additional features.
- Experiment with other machine learning algorithms to compare performance.
- Implement cross-validation to ensure consistent performance.
- Test the spam filter on real-world data and consider deployment options.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib

## Usage

1. Clone the repository.
2. Download the dataset from the UCI Machine Learning Repository.
3. Run the Python script to train the model and classify new messages.

# Quora Question Similarity Measure

This project aims to provide a comprehensive solution for measuring the similarity between questions on Quora. It helps users identify duplicate questions and provides them with relevant answers.

Ckeck the project  : https://quora-question-similarity.onrender.com

## Features

- **Question Similarity Measures**: The website leverages various similarity measures, such as `fuzz_ratio`, `fuzz_partial_ratio`, and `token_sort_ratio`, to determine the similarity between two questions.
- **Duplicate Question Detection**: The website addresses the Quora Question Pairs Similarity Problem by pairing up duplicate questions from Quora. It predicts whether a pair of questions are duplicates or not, allowing it to instantly provide answers to questions that have already been answered.

## Implementation

To implement this idea, you can consider using machine learning algorithms and natural language processing techniques. Some potential approaches include TF-IDF based cosine similarity, logistic regression, support vector machines (SVM), and deep learning methods. These algorithms can help detect similarity in questions asked and provide accurate results.

Remember, the accuracy of the model is crucial for a similarity task. With proper training and evaluation, you can achieve high accuracy, even with a smaller training set. Additionally, interpretability is partially important, allowing users to understand how the similarity measure works.

## Requirements

To run this project, you need the following Python libraries:

```plaintext
fuzzywuzzy==0.18.0
nltk==3.6.5
pandas==1.3.4
scikit-learn==1.0.1
```

Acknowledgments
This project was inspired by the need to identify duplicate questions on Quora.
Special thanks to CampusX for their guidance and support


Feel free to customize this template according to your project's requirements. Good luck with your project!


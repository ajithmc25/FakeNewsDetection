# Week 5

### Literature Review

Papers reviewed:

5. Evaluating Machine Learning Algorithms for Fake News Detection, Shlok Gilda
6. Weakly Supervised Learning for Fake News Detection on Twitter, Stefan Helmstetter and Heiko Paulheim

#### Evaluating Machine Learning Algorithms for Fake News Detection

This paper explores the application of natural language processing techniques for the detection of 'fake news', that is, misleading news stories that come from non-reputable sources. Using a dataset obtained from Signal Media and a list of sources from OpenSources.co, we apply term frequency-inverse document frequency (TF-IDF) of bi-grams and probabilistic context free grammar (PCFG) detection to a corpus of about 11,000 articles. We test our dataset on multiple classification algorithms - Support Vector Machines, Stochastic Gradient Descent, Gradient Boosting, Bounded Decision Trees, and Random Forests. Stochastic Gradient Descent models trained on the TF-IDF feature set performed best.

#### Weakly Supervised Learning for Fake News Detection on Twitter

A practical approach for treating the identification of fake news on Twitter as a binary machine learning problem. While that translation to a machine learning problem is rather straight forward, the main challenge is to gather a training dataset of suitable size. Here, instead of creating a small, but accurate handlabeled dataset, using a large-scale dataset with inaccurate labels yields very good results as well. Achieved an F1 score of 0.77 when only taking into account a tweet as such, and up to 0.9 when also including information about the user account.

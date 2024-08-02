# Project Title:
Coarse-Grained Word Sense Disambiguation (WSD)


# Overview:
This repository is dedicated to the event detection task, which constitutes my **Homework 1** for the course **MULTILINGUAL NATURAL LANGUAGE PROCESSING**, taught by Professor [Roberto Navigli](https://www.diag.uniroma1.it/navigli/), as part of my Master’s in AI and Robotics at [Sapienza University of Rome](https://www.uniroma1.it/it/pagina-strutturale/home). The project explores two distinct approaches to Coarse-Grained WSD:

1. **Baseline Model (Bidirectional LSTM with GloVe pre-trained embeddings):** This approach evaluates the performance of a Bidirectional LSTM model enhanced with GloVe pre-trained word embeddings, providing a foundation for understanding the baseline capabilities in coarse-grained WSD.

2. **Transformer Architecture (RoBERTa):** This method leverages the RoBERTa model, a robust transformer-based architecture, to explore advanced techniques in coarse-grained WSD and compare its performance against the baseline model.

> [!IMPORTANT]
> For Homework 1 on Event Detection as a part of my Multilingual NLP course, please visit the [Event Detection](https://github.com/Sameer-Ahmed7/Event-Detection/) Repository.
>

# Homonym:
In human language, words are often used in several contexts. For distinct Natural Language Processing Applications, It is essential to understand the language's various usage patterns. The same word might have numerous meanings depending on the context. These “difficult” words are more generally referred to as **homonyms.**

## For Example:
Let's take the word _**"bark"**_ as an example:

1. “Cinnamon comes from the bark of the Cinnamon tree.”
2. “The dog barked at the stranger.”

Both sentences use the word _“bark”_, but sentence 1 refers to the outer covering of the tree. While sentence 2 refers to the sound made by a dog. 

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/Homonym.png" width="50%" height="50%" title="Homonym">
  </p>

Therefore, it is clear that the same words can have multiple meanings depending on how they are used in a given sentence. A lot of a word's meaning is defined by how it is used. However, the issue is that when working with text data in NLP, we need a way to interpret the different word's different meanings. To solve this issue WSD (Word Sense Disambiguation) comes into play.


# What is Word Sense Disambiguation (WSD)?

Determining the correct meaning or _"sense"_ of a word in a specific context is a challenge in Natural Language Processing (NLP), and it is known as word sense disambiguation (WSD). Since many words in natural language have numerous meanings, it can be difficult to determine the intended meaning. WSD aims to resolve this issue.

## For Example:
To better understand the challenge of WSD, let's consider the phrase _"light bulb"_ Without any context, it could refer to an object that produces light when connected to electricity. However, with additional context, the meaning can change. 
In the sentence "He had a brilliant idea; a light bulb went off in his head," the term "light bulb" is used metaphorically to represent a sudden understanding or realization. The context surrounding the phrase disambiguates its sense.


# Coarse-Grained vs Fine-Grained:
In the context of Word Sense Disambiguation (WSD), coarse-grained and fine-grained approaches refer to the level of detail in distinguishing between different senses of a word.

## Coarse-Grained:
A coarse-grained approach involves distinguishing between broader, more general senses of a word. This approach groups similar senses into larger categories, focusing on major sense distinctions rather than detailed nuances. It is useful when the goal is to achieve general understanding or when resources are limited.

### Example:

For the word "bank," a coarse-grained approach might categorize its senses into general types like "financial institution" and "side of a river," without differentiating between specific types of financial institutions or riverbanks.

> [!NOTE]
> In our project, we utilize a coarse-grained dataset, which means that our focus is on identifying broad sense categories rather than specific nuances. This choice aligns with the project's goals and constraints, enabling us to develop a system that performs well with a general understanding of word senses.

## Fine-Grained:
A fine-grained approach, in contrast, aims to identify more specific and detailed senses of a word. This approach involves distinguishing between subtle variations and specific meanings, providing a more precise understanding of word usage. It is beneficial for applications requiring detailed semantic distinctions.

### Example:
For the word "bank," a fine-grained approach would differentiate between various types of financial institutions (e.g., "investment bank," "commercial bank") and specific types of riverbanks (e.g., "muddy bank," "rocky bank"), capturing detailed nuances of each sense.

# Data Preparation:
Here I am using the coarse-grained file to solve WSD. In a coarse-grained file, we have **different inputs** _(instance_ids, words, lemmas, pos_tags, and candidates)_ and the **output** is _(senses)_. 

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/data_prep_1.png" width="50%" height="50%" title="Data Preperation 1">
  </p>

To make a program as much as simple, I only take **inputs as (words)**, and **outputs as (senses)**.

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/data_prep_2.png" width="50%" height="50%" title="Data Preperation 2">
  </p>

# Flow Diagram:
This flow diagram illustrates the complete process of the WSD task.
<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/flow_diagram.png" width="50%" height="50%" title="Complete Process of WSD Task">
  </p>

# Model Approach:
In this task, I am considering two different approaches:
1. Baseline Model (Bidirectional LSTM + GloVe pre-trained embedding)
2. Transformer Architecture (RoBERTa)

## 1. Baseline Model:
The approach described utilizes a Bidirectional Long Short-Term Memory (Bi-LSTM) network architecture to capture contextual information from both preceding and subsequent words in a sentence. The model consists of two LSTM layers, with one processing the input sequence forward and the other processing it backward. Pre-trained GloVe embeddings, which provide dense vector representations of words based on co-occurrence statistics, are used as input for the Bi-LSTM. The GloVe embeddings have 300 dimensions and are kept fixed during training. To prevent overfitting, a dropout of 0.4 is applied to the Bi-LSTM's output. Finally, a fully connected layer with softmax activation is used to classify the data.

### Preprocessing Data:

1. Most of the words are in different cases, So I convert all the words into the same case (lower). 
2. Then we need to make a list of both words and senses of training data. It will help us to convert into numbers and for out-of-vocabulary (OOV) problem.
3. The model aims to classify words into senses, but the lengths of the words and senses don't match to address this, a dummy sense key 'PADDING' is created for words that are not homonyms. Padding technique is applied to handle variable length inputs, using the same 'PADDING' key for both inputs and outputs.
4. The use of the same key helps in the loss function (categorical cross-entropy) by ignoring only one key instead of two. This approach allows the model to focus on learning sense keys and not the 'PADDING' sense key.
5. Words and senses are converted into numbers using pre-built lists generated from the training data, resolving out-of-vocabulary (OOV) issues.

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/preprocessing_data_baseline.png" width="50%" height="50%" title="Preprocessing Data Baseline Model">
  </p>

### Flow Diagram of the Model:
This flow diagram illustrates the complete process of the Baseline Model.
<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/flow_diagram_baseline.png" width="30%" height="30%" title="Flow Diagram Baseline Model">
  </p>

## 2. Transformer Architecture (RoBERTa) Model:
The second model, here I am using is the Robustly Optimized BERT Approach (RoBERTa) Transformer, Although Bidirectional LSTM (BiLSTM) models are effective at sequentially capturing contextual dependencies by processing input in both forward and backward directions. 
The reason behind that is in WSD, BiLSTM with pre-trained word embedding (Glove) gives 'same’ embedding for the same words. Although the contexts of words are different, Suppose I have two sentences: "He didn't receive fair treatment" and "Funfair in New York City this summer. In both sentences, the word 'fair' has two different meanings according to the context, but (Glove) embedding gives both 'fair' words embedding the same. The transformer-based model RoBERTa, on the other hand, is capable of effectively capturing contextual data. Transformers, like RoBERTa, use self-attention mechanisms to recognize relationships between words in a sentence, which improves their ability to effectively recognize contextual information. 

### Preprocessing Data:
1. Most of the words are in different cases, So I convert all the words into the same case (lower). 
2. Then we need to make a list of both words and senses of training data. It will help us to convert into numbers and for out-of-vocabulary (OOV) problem.
3. Inputs (words) are in lowercase and need to be tokenized. Tokenization is crucial for RoBERTa models, and the 'roberta-base' tokenizer is used. 'roberta-base' utilizes subword tokenization (BPE variant) that divides words into subword units based on frequency. Tokenization is applied to inputs using the Datasets class.
4. Tokenization changes the length of the output (senses). An align_label() function is used to assign keys only to homonyms, while non-homonyms are assigned 'PADDING' value (-100). This maintains the length of the sense key list for outputs. 
5. The -100 values are ignored during training with categorical cross-entropy loss.
<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/preprocessing_data_roberta.png" width="50%" height="50%" title="Preprocessing Data RoBERTa Model">
  </p>
  
### Flow Diagram of the Model:
This flow diagram illustrates the complete process of the RoBERTa Model.
<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD/blob/main/assets/flow_diagram_roberta.png" width="30%" height="30%" title="Flow Diagram RoBERTa Model">
  </p>

# Results:

| Models                                                 | Accuracy (Validation Data) | Accuracy (Test Data) |
|--------------------------------------------------------|----------------------------|----------------------|
| Baseline Model (Bi-LSTM + Pretrained word embedding – Glove) | 72.7%                      | 71.2%                |
| RoBERTa Transformer                                    | 90.1%                      | 89.0%                |


# Conclusion
The Coarse-Grained Word Sense Disambiguation (WSD) project has successfully highlighted the effectiveness of modern transformer models in improving WSD tasks. Our baseline model, which utilized a Bi-LSTM architecture with pre-trained GloVe embeddings, provided a solid starting point with an accuracy of **71.2%** on the test data. However, the RoBERTa Transformer significantly outperformed the baseline, achieving an impressive **89.0%** accuracy. This stark improvement underscores the superior capability of transformer-based models in capturing contextual nuances in language, making them highly suitable for WSD applications. Future work will explore further optimizations and potential integrations with other advanced NLP frameworks.













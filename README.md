# Project Title:
Coarse-Grained Word Sense Disambiguation (WSD)


# Overview:
This repository is dedicated to the event detection task, which constitutes my **Homework 1** for the course **MULTILINGUAL NATURAL LANGUAGE PROCESSING**, part of my Master’s in AI and Robotics at [Sapienza University of Rome](https://www.uniroma1.it/it/pagina-strutturale/home). The project explores two distinct approaches to Coarse-Grained WSD:

1. **Baseline Model (Bidirectional LSTM with GloVe pre-trained embeddings):** This approach evaluates the performance of a Bidirectional LSTM model enhanced with GloVe pre-trained word embeddings, providing a foundation for understanding the baseline capabilities in coarse-grained WSD.

2. **Transformer Architecture (RoBERTa):** This method leverages the RoBERTa model, a robust transformer-based architecture, to explore advanced techniques in coarse-grained WSD and compare its performance against the baseline model.

> [!IMPORTANT]
> For Homework 1 on Event Detection as a part of my Multilingual NLP course, please visit the [Event Detection](https://github.com/Sameer-Ahmed7/Event-Detection/) Repository.
>

# Homonyms:
In human language, words are often used in several contexts. For distinct Natural Language Processing Applications, It is essential to understand the language's various usage patterns. The same word might have numerous meanings depending on the context. These “difficult” words are more generally referred to as **homonyms.**

## For Example:
Let's take the word _**"bark"**_ as an example:

1. “Cinnamon comes from the bark of the Cinnamon tree.”
2. “The dog barked at the stranger.”

Both sentences use the word _“bark”_, but sentence 1 refers to the outer covering of the tree. While sentence 2 refers to the sound made by a dog. 

Therefore, it is clear that the same words can have multiple meanings depending on how they are used in a given sentence. A lot of a word's meaning is defined by how it is used. However, the issue is that when working with text data in NLP, we need a way to interpret the different word's different meanings. To solve this issue WSD (Word Sense Disambiguation) comes into play.


# What is Word Sense Disambiguation (WSD)?

Determining the correct meaning or _"sense"_ of a word in a specific context is a challenge in Natural Language Processing (NLP), and it is known as word sense disambiguation (WSD). Since many words in natural language have numerous meanings, it can be difficult to determine the intended meaning. WSD aims to resolve this issue.

## For Example:
To better understand the challenge of WSD, let's consider the phrase _"light bulb"_ Without any context, it could refer to an object that produces light when connected to electricity. However, with additional context, the meaning can change. 
In the sentence "He had a brilliant idea; a light bulb went off in his head," the term "light bulb" is used metaphorically to represent a sudden understanding or realization. The context surrounding the phrase disambiguates its sense.




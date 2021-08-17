## 1.Problem Statement, Data and Metric Analysis
---

### **A**.**Problem Statement :** 

The Aim of this case study is to build a Good RNN model that corrects the Noised, Non-Grammatical and mispelled social media comments to a better addressed, grammatically strong and rule based english coded sentences.

As we know while using social media, We have seen bunch of half-short forms of english words and sentences. So, to use RNN in state-of-the art Natural Language Processsing for Translation of sentences, we have tried approach.

This case study is based on Research Paper: Sentence Correction using Recurrent Neural Networks by:

Gene Lewis,Department of Computer Science

Stanford University
Stanford

CA 94305

Link: http://cs224d.stanford.edu/reports/Lewis.pdf


### **B**.**How Deep Learning will help solving this problem?**

As we have seen Sequence to Sequence translation of texts/sentences to predict possible words, Taking an example of search engine, You have seen possible suggestions over engine when you type some set of words.

Hence, to use that kind of  Deep learning technique to build a model,to correctly present or predict the correct translation of sentences, we have headed off with this research paper.

This paper describes how character level and word level embeddings can be used as our inputs and thus, predicting right asnwers.

There are 2 parts for this, Encoder and Decoder models. That we will see in the future approach.

###**C**.**Data Definition :** 

Data is taken from English corpus of 2000 texts from the National University of Singapore https://www.comp.nus.edu.sg/~nlp/corpora.html. 


NUS Social Media Text Normalization and Translation Corpus:

The corpus is created for social media text normalization and translation. It is built by randomly selecting 2,000 messages from the NUS English SMS corpus. The messages were first normalized into formal English and then translated into formal Chinese.

**Example data from the testing set:**

Input:  "Ic...Haiz,nv ask me along?Hee,im so sian at hm."

Output: "I see. Sigh, why do you never ask me along? I’m so bored at home."

### **D**.**Metric Used :**

There are three types of Metrics that is used to check our text similarities:

1. Word and Character Based Perplexities

2. Word and Character Based Bleu Score

3. NIST Score
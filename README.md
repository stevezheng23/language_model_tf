# Language Model
Language modeling is a task that assigns probabilities to sequences of words or various linguistic units (e.g. char, subword, sentence, etc.). Language modeling is one of the most important problem in modern natural language processing (NLP) and it's used in many NLP applications (e.g. speech recognition, machine translation, text summarization, spell correction, auto-completion, etc.). In the past few years, neural approaches have achieved better results than traditional statistical approaches on many language model benchmarks. Moreover, recent work has shown language model pre-training can improve many NLP tasks in different ways, including feature-based strategies (e.g. ELMo, etc.) and fine-tuning strategies (e.g. OpenAI GPT, BERT, etc.), or even in zero-shot setting (e.g. OpenAI GPT-2, etc.).

<img src="/language_model/document/language_model.example.png" width=600><br />
*Figure 1: An example of auto-completion powered by language modeling*

## Setting
* Python 3.6.6
* Tensorflow 1.12
* NumPy 1.15.4
* NLTK 3.3

## DataSet
* The [Wikipedia corpus](https://www.corpusdata.org/wikipedia.asp) contains about 2 billion words of text from a 2014 dump of the Wikipedia (about 4.4 million pages). As far as we are aware, our Wikipedia full-text data is the only version available from a recent copy of Wikipedia.
* [BooksCorpus](http://yknzhu.wixsite.com/mbweb): Books are a rich source of both fine-grained information, how a character, an object or a scene looks like, as well as high-level semantics, what someone is thinking, feeling and how these states evolve through a story. This work aims to align books to their movie releases in order to provide rich descriptive explanations for visual content that go semantically far beyond the captions available in current datasets.
* The purpose of [1 Billion Word](http://www.statmt.org/lm-benchmark/) benchmark is to make available a standard training and test setup for language modeling experiments. This benchmark contains almost one billion words of training data, and it's aiming to help researcher to quickly evaluate novel their language modeling techniques, and to easily compare the contributions when combined with other advanced techniques.
* [GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Usage
* Run experiment
```bash
# run experiment in train + eval mode
python language_model_run.py --mode train_eval --config config/config_lm_template.xxx.json
# run experiment in train only mode
python language_model_run.py --mode train --config config/config_lm_template.xxx.json
# run experiment in eval only mode
python language_model_run.py --mode eval --config config/config_lm_template.xxx.json
```
* Visualize summary
```bash
# visualize summary via tensorboard
tensorboard --logdir=output
```
* Encode text
```bash
# encode text into vector
python language_model_run.py --mode encode --config config/config_lm_template.xxx.json
```
## Reference
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee,
and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.[Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
* Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. [Aligning books and movies: Towards story-like visual explanations by watching movies and reading books](https://arxiv.org/abs/1506.06724) [2015]
* Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. [One billion word benchmark for measuring progress in statistical language modeling](https://arxiv.org/abs/1312.3005) [2013]

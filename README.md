# LongPegasus


## A Longer Version of Pegasus TF Model For Abstractive Summarization :robot:

![img1](https://miro.medium.com/max/1184/1*yp5xLaVL7vOs6YT9QPO2Ow.png)


This package is used for inducing longformer self attention over base pegasus abstractive summarization model to increase the token limit and performance.The Pegasus is a large Transformer-based encoder-decoder model with a new pre-training objective which is adapted to abstractive summarization. More specifically, the pre-training objective, called "Gap Sentence Generation (GSG)", consists of masking important sentences from a document and generating these gap-sentences.On the other hand, the Longformer is a Transformer which replaces the full-attention mechanism (quadratic dependency) with a novel attention mechanism which scale linearly with the input sequence length. Consequently, Longformer can process sequences up to 4,096 tokens long (8 times longer than BERT which is limited to 512 tokens).This package plugs Longformers attention mechanism to Pegasus in order to perform abstractive summarization on long documents. The base modules are built on Tensorflow platform.

## Contributing

Pull requests are (not yet)welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT

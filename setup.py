# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:30 2021

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'LongPegasus',         
  packages = ['LongPegasus'],   
  version = '0.1',       
  license='MIT',        
  description = 'A Longer Version of Pegasus TF Model For Abstractive Summarization',   
  long_description='This package is used for inducing longformer self attention over base pegasus abstractive summarization model to increase the token limit and performance.The Pegasus is a large Transformer-based encoder-decoder model with a new pre-training objective which is adapted to abstractive summarization. More specifically, the pre-training objective, called "Gap Sentence Generation (GSG)", consists of masking important sentences from a document and generating these gap-sentences.On the other hand, the Longformer is a Transformer which replaces the full-attention mechanism (quadratic dependency) with a novel attention mechanism which scale linearly with the input sequence length. Consequently, Longformer can process sequences up to 4,096 tokens long (8 times longer than BERT which is limited to 512 tokens).This package plugs Longformers attention mechanism to Pegasus in order to perform abstractive summarization on long documents. The base modules are built on Tensorflow platform.',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/LongPegasus',   
  download_url = 'https://github.com/abhilash1910/LongPegasus/archive/v_01.tar.gz',    
  keywords = ['Longformer','Self Attention','Global Attention','Gap sentence generation','Pegasus','Transformer','Encoder Decoder','Tensorflow'],   
  install_requires=[           

          'tensorflow',
          'transformers',
          'sentencepiece'
          
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',      
    'Programming Language :: Python :: 3.8',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 00:06:23 2021

@author: Abhilash
"""

import transformers
from transformers.models.longformer.modeling_tf_longformer import TFLongformerModel,TFLongformerSelfAttention,TFLongformerSelfOutput
import tensorflow as tf
from typing import List, Optional, Tuple, Dict
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration
from transformers.models.pegasus.configuration_pegasus import PegasusConfig
#from transformers.models.pegasus.modeling_tf_pegasus import shift_tokens_right


class LongformerForPegasus(TFPegasusForConditionalGeneration):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        for j,layer in enumerate(self.model.encoder.layers):
            layer.self_attn=TFLongformerSelfAttention(config=config,layer_id=j)

            
class LongformerPegasusConfig(PegasusConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,initializer_range=None,layer_norm_eps=1e-5,
                 hidden_dropout_prob=1e-5,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range=initializer_range
        self.layer_norm_eps=layer_norm_eps
        self.hidden_dropout_prob=hidden_dropout_prob
        assert self.attention_mode=='sliding_chunks'
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']            
            
class LongformerSelfAttentionPegasus(tf.keras.layers.Layer):
    def __init__(self,config,layer_id,**kwargs):
        super().__init__(**kwargs)
        self.self_attention=TFLongformerSelfAttention(config,layer_id,name="self")
        self.outputs=TFLongformerSelfOutput(config,name="outputs")
    def call(self,inputs,training=False):
        (hidden_states,attention_mask,layer_head_mask,is_index_masked,is_index_global_attn,is_global_attn)= inputs
        self_outputs=self.self_attention([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn],
            training=training,)
        attention_output=self.outputs(self_outputs[0],hidden_states,training=training)
        if len(self_outputs==2):
            ops=(attention_output,)+self_outputs[1:]
        else:
            ops=(attention_output,None)
        return ops
               
               
class LongPegasus():
    def create_long_model(self,save_model,attention_window,max_pos,model_name):
        if model_name==None:
            self.base_model='google/pegasus-xsum'
        else:
            self.base_model=model_name
        self.model=TFPegasusForConditionalGeneration.from_pretrained(self.base_model)
        self.tokenizer=PegasusTokenizer.from_pretrained(self.base_model,model_max_length=max_pos)
        self.configuration=LongformerPegasusConfig.from_pretrained(self.base_model)
        self.model.config=self.configuration
        self.configuration.attention_probs_dropout_prob=self.configuration.attention_dropout
        self.configuration.architectures= ['LongformerForPegasus']
        self.tokenizer.model_max_length=max_pos
        self.tokenizer.init_kwargs['model_max_length']=max_pos
        current_max_pos, embed_size = self.model.model.encoder.embed_positions.weight.shape
        assert current_max_pos == self.configuration.max_position_embeddings

        self.configuration.max_encoder_position_embeddings = max_pos 
        self.configuration.max_decoder_position_embeddings = self.configuration.max_position_embeddings #512
        print("max_encoder_position_embeddings: ", self.configuration.max_encoder_position_embeddings)
        print("max_decoder_position_embeddings: ", self.configuration.max_decoder_position_embeddings)
        assert max_pos>current_max_pos
        new_encoder_positional_embedded=self.model.model.encoder.embed_positions.weight
        temp=tf.random.normal(shape=(max_pos,embed_size))
        new_encoder_positional_embed=tf.Variable(temp)
        #new_encoder_positional_embed=tf.tensor(max_pos,embed_size)
        #bart=2
        #pegasus=0
        k=0
        step=current_max_pos-k
        while(k<max_pos-1):
            #step loop over to increase embedding
            new_encoder_positional_embed[k:(k+step)].assign(self.model.model.encoder.embed_positions.weight[:])
            k+=step
        self.model.model.encoder.embed_positions=new_encoder_positional_embed
        self.configuration.attention_window = [attention_window] * self.configuration.num_hidden_layers
        self.configuration.attention_dilation = [1] * self.configuration.num_hidden_layers
        for i, layer in enumerate(self.model.model.encoder.layers):
            longformer_self_attn_for_pegasus = LongformerSelfAttentionPegasus(self.configuration, layer_id=i)
            layer.self_attn = longformer_self_attn_for_pegasus
        print("=======Model Configuration=======")
        print(self.model.config)
        self.model.save_pretrained(save_model)
        self.tokenizer.save_pretrained(save_model)
        print("==============Model & Tokenizer saved===============")
        return self.model, self.tokenizer
        
               
               

 
        

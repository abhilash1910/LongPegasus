# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 00:25:36 2021

@author: Abhilash
"""


from LongPegasus.LongPegasus import LongPegasus
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration

if __name__=='__main__':
    l=LongPegasus()             
    model_name=None            
    model,tokenizer=l.create_long_model(save_model="E:\\Pegasus\\", attention_window=4096, max_pos=4096,model_name=model_name)
    model = TFPegasusForConditionalGeneration.from_pretrained('E:/Pegasus/')
    tokenizer = PegasusTokenizer.from_pretrained('E:/Pegasus/')

    ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=4096, return_tensors='tf')
    
    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'])
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
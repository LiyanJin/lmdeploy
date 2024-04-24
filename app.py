import gradio as gr
import os
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

base_path = './lmdeploy'
os.system(f'git clone https://code.openxlab.org.cn/LiyanJin/lmdeploy.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

# tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True,load_in_4bit=True, device_map="auto").cuda()

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2,model_format="awq")

pipe = pipeline(base_path,backend_config=backend_config)
gen_config = GenerationConfig(top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=1024)

def chat(message,history):
    response = pipe(message, model_name='internlm2-chat-1_8b-4bit', gen_config = gen_config)
    return response.text

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8b-4bit",
                description="""
InternLM2-4bit is mainly developed by jin.  
                 """,
                 ).queue(1).launch()


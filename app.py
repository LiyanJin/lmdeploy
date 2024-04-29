import os
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig

backend_config = TurbomindEngineConfig(model_format="awq")

# download internlm2 to the base_path directory using git tool
base_path = './lmdeploy'
os.system(f'git clone https://code.openxlab.org.cn/LiyanJin/lmdeploy.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

model_name = 'internlm2-chat-1_8b-4bit'
pipe = pipeline(base_path,model_name, backend_config=backend_config)
gen_config = GenerationConfig(top_p=0.8,top_k=40,temperature=0.8,max_new_tokens=1024)

def chat(message,history):
  response = pipe(message, gen_config = gen_config)
  return response.text

demo = gr.ChatInterface(
  fn = chat,
  title="InternLM2-Chat-1.8b-4bit",
  description="""InternLM2-4bit is mainly developed by jin. """,
)
demo.launch()

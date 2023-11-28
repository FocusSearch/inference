from transformers import  AutoTokenizer,AutoModelForCausalLM
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import pandas as pd
import csv
import ast
import argparse
import re
# import torch
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='Qwen/Qwen-7B-Chat',help='model')
parser.add_argument('--lora_path',default=None,help='lora_path')
parser.add_argument('--data_path',default='benchmark_data.xlsx',help='data_path')
parser.add_argument('--out_path',default='output.csv',help='output_dir')
args = parser.parse_args()
model_path = args.model_path
data_path = args.data_path
out_path = args.out_path
lora_path = args.lora_path
raw_data = pd.read_excel(data_path)
out_answer = []

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if args.lora_path:
    model = AutoPeftModelForCausalLM.from_pretrained(
    lora_path, # path to the output directory
    device_map="auto",
    trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()


# 可指定不同的生成长度、top_p等相关超参
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

for _,data_item in raw_data.iterrows():
    question = data_item['question']
    table = data_item['schema']
    schema = ast.literal_eval(table)
    print(schema)
    context = f"你是一个经验丰富的数据分析师。根据数据表结构(schema)，你需要将用户的数据查询描述(question)翻译成特定的DSL。\"\"\"schema\"\"\"为数据表结构，包括列名和列的属性，###question###为用户的查询描述。\nInput: schema:\"\"\"{schema}\"\"\"\n---\nquestion:###{question}###"
    answer, _ = model.chat(tokenizer, context, history=None)

    # ids = tokenizer.encode(context)
    # input_ids = torch.LongTensor([ids])
    # input_ids = input_ids.to(0)
    # out = model.generate(
    #         input_ids=input_ids,
    #         max_new_tokens=64,
    #         #repetition_penalty=1.2,
    #         #do_sample=True,
    #         temperature=0.01
    #     )
    #answer = tokenizer.decode(out[0])
    print(answer)
    out_answer.append(answer)

out = pd.DataFrame({'question':raw_data['question'],'answer':out_answer})
out.to_csv(out_path,index=False,sep='|',encoding='utf_8_sig',quoting=csv.QUOTE_NONE,escapechar='|')
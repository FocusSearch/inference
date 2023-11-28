# 指导
## 1. 说明
inference.py是Qwen系列LoRA微调后，针对benchmark数据的推理脚本。
修改infer.sh中的model_path为LLM的权重路径，修改lora_path为微调后的LoRA权重路径。
若没有指定LoRA路径，则直接读取model_path路径下的模型权重，可用于权重合并后的benchmark推理场景。

## 2. 依赖安装
requirements.txt中transformers==4.32.0和transformers_stream_generator==0.0.4为Qwen 官方实现的推荐版本，其它部分版本应当也可以。
可额外选择安装编译flash-attention提高运行效率以及降低显存占用，不安装也不影响运行:
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .

## 3. 运行
centos环境下执行 sh infer.sh

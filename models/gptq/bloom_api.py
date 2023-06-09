import time

import torch
import torch.nn as nn
import argparse
from gptq import *
from modelutils import *
from quant import *
import logging
from flask import Flask, request, jsonify, abort
from transformers import AutoTokenizer
app = Flask(__name__)

def get_bloom(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize):
    from transformers import BloomConfig, BloomForCausalLM
    config = BloomConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = BloomForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print(f'加载模型：{checkpoint}')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


def load_model(model_name, load_local, wbits=16, groupsize=-1):
    model = load_quant(model_name, load_local,wbits, groupsize)
    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def do_predict(model, tokenizer, input_text, min_length=10, max_length=1024, top_p=0.95, temperature=0.8, only_anwser=True):
    inputs = 'Human: ' + input_text.strip() + '\n\nAssistant:'
    input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    input_length = len(inputs)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode([el.item() for el in generated_ids[0]])
    if only_anwser:
        result = result[input_length:]
    return result


@app.route("/api/predict", methods=['POST'])
def predict():
    """
    聊天预测
     curl -d '{"data":"你好"}' -H "Content-Type: application/json" -X POST http://192.168.50.209:5303/api/predict
    Args:
        data = "content"
    Returns: 返回格式
     嵌套列表 预测的返回的结果，所有可能关系
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    if not data:
        return jsonify({"code": 400, "msg": "data不能为空"}), 400
    logging.info(f"数据分别是: {data}")
    result = do_predict(model, tokenizer, data, args.min_length, args.max_length, args.top_p, args.temperature)
    logging.info(f"预测结果是: {result}")
    return jsonify(result)

@app.route("/ping", methods=['GET', 'POST'])
def ping():
    """
    测试
    :return:
    :rtype:
    """
    return jsonify("Pong")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--max_length', type=int, default=1024,
        help='The maximum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--top_p', type=float, default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )

    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )
    # 添加device参数
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='The device to use for inference.'
    )
    args = parser.parse_args()
    if args.device == 'cpu':
        DEV = torch.device('cpu')
    else:
        DEV = torch.device('cuda:0')
    # 调用模型
    model, tokenizer = load_model(args.model, args.load, args.wbits, args.groupsize)
    # 推理
    app.run(host='0.0.0.0', port=5303, debug=False, threaded=True)

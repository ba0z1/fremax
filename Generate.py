import os
import argparse
import logging
import utils
import random
import math
import json
import jsonlines
from configparser import ConfigParser
from encode import encoder, encoder_configs
from decode import decoder, decoder_configs



# def parse_arg_main():
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="Train or Generate", choices=["T", "G", "E"], default="G")
parser.add_argument("--gpuid", type=str, default="1")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="/data3/ADG_P/redditlm/softmax_with_frequency/movie/30ep")
parser.add_argument("--generate_config", type=str,
                    default="/data3/ADG_P/redditlm/configs/generate_plain.json")
parser.add_argument("--model_name_or_path", type=str, default="/data3/ADG_P/models_trained")
# parser.add_argument("--model_name_or_path", type=str, default="/data/RedditLM/bloomz-560m-lora")
parser.add_argument("--base_model_name_or_path", type=str, default="/data3/ADG_P/models_trained")
# parser.add_argument("--base_model_name_or_path", type=str, default="bigscience/bloomz-560m")
parser.add_argument("--deepspeed", type=str, help="deepspeed config")
# parser.add_argument("--lora_hyperparams_file", default="./configs/lora_config.json", type=str, help="Provide it when use_lora=True")
parser.add_argument("--lora_hyperparams_file",type=str,
                    help="Provide it when use_lora=True")
parser.add_argument("--use_lora", action="store_true", default=False, help="Use lora")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--T", type=float,default=1.0)
parser.add_argument("--P", type=float,default=1.0)
parser.add_argument("--theta", type=float,default=0.01) # 熵因子
parser.add_argument("--alpha", type=float,default=1.2) # 乘方因子
parser.add_argument("--beta", type=float,default=2)#bias of log()
parser.add_argument("--cache_dir", type=str, default="/data3/ADG_P/huggingface")
    # return parser.parse_args(["--use_lora"])


args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from peft import (
# PeftModel
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)


def loadprompt(infile_path):
    '''
    load your own prompt template
    '''
    prompts = []

    # with open(infile_path, "r") as f_in:
    #     while True:
    #         line = f_in.readline()
    #         if not line:
    #             break
    #         items = json.loads(line)
    #         input_sentence = items["input"]
    #         instruction = items["instruction"]
    #         prompts.append(f"### Text:\n{input_sentence}\n\n### Comment:\n" if input_sentence == "" else
    #                        f"### Text:\n{instruction}\n{input_sentence}\n\n### Comment:\n")
    return prompts


def load_model_and_tokenizer(args, use_lora=False, is_rnn=False, load_in_8bit=False):
    model_name_or_path = args.model_name_or_path if not use_lora else args.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              cache_dir=os.path.join(args.cache_dir, "hub"))
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                              load_in_8bit=load_in_8bit,
                                              device_map="auto",
                                              cache_dir=os.path.join(args.cache_dir, "hub"))
    # TODO
    #   load RNN without/with transformers
    # if use_lora:
    #     model = PeftModel.from_pretrained(
    #         model,
    #         args.model_name_or_path,
    #         torch_dtype="auto"
    #     )
    if is_rnn:
        pass
    return tokenizer, model


def get_probs(model, input_ids, past_key_and_values=None, forbiddenids=None, **kwargs):
    if past_key_and_values is None:
        logits = model(input_ids).logits[0, -1, :]
        probs = logits.softmax(dim=-1)
        if forbiddenids is not None:
            probs[forbiddenids] = 0
        probs = probs/probs.sum()
        return probs
    else:
        pass
def get_probs_without_softmax(model, input_ids, past_key_and_values=None, forbiddenids=None, **kwargs):
    if past_key_and_values is None:
        logits = model(input_ids).logits[0, -1, :]
        probs = logits / logits.sum()

        # probs = logits.softmax(dim=-1)
        if forbiddenids is not None:
            probs[forbiddenids] = 0
        probs = probs/probs.sum()
        return probs
    else:
        pass
def get_probs_with_temperature(model, input_ids, past_key_and_values=None, forbiddenids=None, temperature=1.0,**kwargs, ):
    if past_key_and_values is None:
        logits = model(input_ids).logits[0, -1, :]
        probs = (logits / temperature).softmax(dim=-1)
        if forbiddenids is not None:
            probs[forbiddenids] = 0
        probs = probs/probs.sum()
        return probs
    else:
        pass

def train(args, configs):
    pass


def generate(args, configs):
    os.makedirs(args.out_dir, exist_ok=True)

    '''
    load model and tokenizer
    '''
    tokenizer, model = load_model_and_tokenizer(args, args.use_lora)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:d}".format(total_trainable_params))

    '''
    load bits stream for testing
    '''
    with open(configs["bit_filepath"], 'r', encoding='utf8') as f:
        bit_stream_ori = f.read().strip()
    bit_stream = list(bit_stream_ori)
    bit_stream = ''.join(bit_stream)
    bit_stream = ""
    bit_index = int(torch.randint(0, high=10000, size=(1,)))

    '''
    load generate configs
    '''
    alg = configs["algorithm"]
    max_new_tokens = configs["max_new_tokens"]
    generate_num = configs["generate_num"]
    kwargs = configs[alg]
    outfile = "-".join([os.path.basename(args.model_name_or_path), alg, str(generate_num), str(max_new_tokens)] +
                       [f'T={args.T}', f'P={args.P}'] )

    # prompt = loadprompt(configs["prompt_filepath"])
    try:
        forbidden_ids = json.load(open(configs["forbiddenid_filepath"], "r"))
    except:
        forbidden_file_path = configs["forbiddenid_filepath"]
        print(f"no forbidden id file {forbidden_file_path}")
        forbidden_ids = None
    '''
    load frequency file
    '''
    import numpy as np
    with open("movie_frequency.json","r") as f:
        dic = json.load(f)
    frequency = []
    for i in range(len(dic)):
        frequency.append(dic[str(i)])
    frequency = np.array(frequency)

    
    '''
    start generate
    '''
    model.eval()
    with torch.no_grad():
        stega_text = []
        stega_idx = 0
        with jsonlines.open(os.path.join(args.out_dir, f"alpha={args.alpha}_beta={args.beta}_alg={alg}.jsonl"), "w") as f:
            while len(stega_text) < generate_num:   
                stega_sentence = []
                stega_bit = []
                # try:
                if len(bit_stream[bit_index:]) <= max_new_tokens * math.log2(tokenizer.vocab_size):
                    bit_stream_shuffle = list(bit_stream_ori)
                    random.shuffle(bit_stream_shuffle)
                    bit_stream += "".join(bit_stream_shuffle) # add more bits

                prompt_text = ""
                input_ids = tokenizer.encode(tokenizer.bos_token + prompt_text, return_tensors="pt").to(device)
                if prompt_text == "":
                    probs = get_probs(model, input_ids[:, -1024:], forbidden_ids=forbidden_ids,temperature=args.T)



                    # for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                    #     probs[:, forbidden_id] = 0
                    prev, num_bits_encoded = encoder("plain", probs, bit_stream, bit_index, **configs["plain"]) # sample start word using plain configs
                    # x = torch.cat([input_ids, prev], dim=1)
                    prompt_text = tokenizer.decode(prev[0]).strip()
                    input_ids = tokenizer.encode(tokenizer.bos_token + prompt_text, return_tensors="pt").to(device)
                    x = input_ids
                    stega_sentence += tokenizer.encode(prompt_text)
                    stega_bit += [""] * len(stega_sentence)
                else:
                    x = input_ids



                # for forbidden_id in range(256):
                #     probs[:, forbidden_id] = 0
                if alg.lower() == "ac":
                    max_val = 2 ** kwargs["precision"] # num of intervals ; max_val = 2**52
                    cur_interval = [0, max_val]
                
                for i in range(max_new_tokens - 1):
                    if tokenizer.eos_token_id in stega_sentence:
                        break
                    # conditional probability distribution
                    log_prob = model(x[:, -1024:]).logits[:, -1, :]

                    # get probs
                    log_prob -= log_prob.max()
                    ''' 原始softmax-t
                    prob = torch.exp(log_prob / args.T).reshape(-1) # softmax
                    '''
                    '''熵调整 softmax
                    调整参数theta
                    '''
                    # prob = torch.exp(log_prob).reshape(-1)
                    # entropy = -(prob * torch.log(prob)).sum()
                    # # for i in range(5):
                    # #     if entropy > -(prob * torch.log(prob)).sum() and -(prob * torch.log(prob)).sum() > 10:
                    # #         entropy = -(prob * torch.log(prob)).sum()
                    # #     else:
                    # #         break
                    # t =  (0.8 + args.theta * torch.log(1 + entropy))
                    # # print(t)
                    # prob = torch.exp(log_prob / t).reshape(-1)
                    '''频率化softmax v1
                        调整参数alpha
                    '''
                    
                    # tmp_frequency = frequency.copy()
                    # for i in x[0]:
                    #     tmp_frequency[i] = math.sqrt(tmp_frequency[i])
                    # tmp_frequency = torch.exp(- torch.Tensor(tmp_frequency).to('cuda') / 2588580).reshape(-1)
                    # prob = torch.exp(log_prob).reshape(-1)
                    # entropy = -(prob * torch.log(prob)).sum()
                    # # for i in range(len(prob)):
                    # #     prob[i] = frequency[str(i)] * prob[i]
                    # # for i in range(len(prob)):
                    # prob = prob * (1 - torch.log(1 + torch.pow(tmp_frequency,args.alpha)))
                    # prob = prob - prob.min()
                    # print(x)

                    '''频率化softmax v2
                        直接乘频率
                    '''
                    
                    tmp_frequency = frequency.copy()
                    # 重复惩罚
                    for i in x[0]:
                        tmp_frequency[i] = math.sqrt(tmp_frequency[i])
                    # tmp_frequency = torch.exp(- torch.Tensor(tmp_frequency).to('cuda') / 2588580).reshape(-1)
                    prob = torch.exp(log_prob).reshape(-1)
                    entropy = -(prob * torch.log(prob)).sum()
                    # for i in range(len(prob)):
                    #     prob[i] = frequency[str(i)] * prob[i]
                    # for i in range(len(prob)):
                    prob = prob * (torch.log(args.beta + torch.pow(torch.Tensor(tmp_frequency).to('cuda'), args.alpha)))
                    # prob = prob - prob.min()
                    # print(x)
                    # for forbidden_id in range(256):
                    #     if forbidden_id == tokenizer.eos_token_id:
                    #         continue
                    #     else:
                    #         prob[forbidden_id] = 0
                    prob = prob / prob.sum()

                    # early stop generation
                    if alg.lower() == "ac":
                        cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, top_p = args.P, **kwargs)
                    else:
                        prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)
                    if int(prev) == tokenizer.eos_token_id:
                        break
                    stega_sentence.append(int(prev))
                    x = torch.cat([x, prev], dim=1)
                    stega_bit.append(bit_stream[bit_index:bit_index+num_bits_encoded])
                    bit_index += num_bits_encoded
                if tokenizer.eos_token_id in stega_sentence:
                    stega_sentence.remove(tokenizer.eos_token_id)
                stega_text.append(tokenizer.decode(stega_sentence))
                stega_idx += 1
                f.write({"idx": stega_idx,
                            "prompt": prompt_text,
                            "stego": tokenizer.decode(stega_sentence),
                            "tokens": stega_sentence,
                            "bits": stega_bit})
                # print("idx: ",  stega_idx)
                print("bit_idx: ", bit_index)
                print("prompt: ", prompt_text)
                print("stego: ", tokenizer.decode(stega_sentence))
                print(f"idx={stega_idx},t={args.T}, p={args.P} output_path={outfile}")
            # except Exception as e:
                # print(e)
                # stega_idx += 1
                # print("idx: ",  stega_idx)
                # print("bit_idx: ", bit_index)
                # print("prompt: ", prompt_text)
                # print("stego: ", tokenizer.decode(stega_sentence))
                # print(f"idx={stega_idx},t={args.T}, p={args.P} output_path={outfile}")



def extract(args, configs):
    os.makedirs(args.out_dir, exist_ok=True)
    '''
    load model and tokenizer
    '''
    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, args.use_lora)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:d}".format(total_trainable_params))

    '''
    load generate configs
    '''
    alg = configs["algorithm"]
    max_new_tokens = configs["max_new_tokens"]
    generate_num = configs["generate_num"]
    kwargs = configs[alg]
    infile = "-".join([os.path.basename(args.model_name_or_path), alg, str(generate_num), str(max_new_tokens)] +
                       ["{}{}".format(k, v) for k, v in kwargs.items()])

    try:
        forbidden_ids = json.load(open(configs["forbiddenid_filepath"], "r"))
    except:
        forbidden_file_path = configs["forbiddenid_filepath"]
        print(f"no forbidden id file {forbidden_file_path}")
        forbidden_ids = None

    '''
    start generate
    '''
    model.eval()
    with torch.no_grad():
        stega_text = []
        stega_idx = 0
        with open(os.path.join(args.out_dir, infile + ".jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                prompt_text = item["prompt"]
                stego_text = item["stego"]
                embed_bits = item["bits"]
                input_ids = tokenizer.encode(tokenizer.bos_token + prompt_text, return_tensors="pt").to(device)
                if prompt_text == "":
                    full_ids = tokenizer.encode(tokenizer.bos_token + stego_text, return_tensors="pt").to(device)
                else:
                    full_ids = tokenizer.encode(tokenizer.bos_token + prompt_text + stego_text, return_tensors="pt").to(device)

                # full_logits = model(full_ids).logits
                if alg.lower() == "ac":
                    max_val = 2 ** kwargs["precision"]  # num of intervals ; max_val = 2**52
                    cur_interval = [0, max_val]

                full_bits = []
                for i in range(len(input_ids[0]), len(full_ids[0])):
                    # log_prob = full_logits[:, i-1, :]
                    log_prob = model(full_ids[:, :i]).logits[:, -1, :]
                    log_prob -= log_prob.max()
                    prob = torch.exp(log_prob).reshape(-1)
                    for forbidden_id in range(256):
                        if forbidden_id == tokenizer.eos_token_id:
                            continue
                        else:
                            prob[forbidden_id] = 0
                    prob = prob / prob.sum()
                    # early stop generation
                    if alg.lower() == "ac":
                        cur_interval, extract_bits = decoder(alg, prob, full_ids[0][i], cur_interval, **kwargs)
                    else:
                        extract_bits = decoder(alg, prob, full_ids[0][i], **kwargs)
                    full_bits.append(extract_bits)
                print()


if __name__ == '__main__':
    utils.set_seed(args.seed)
    if args.mode == "T":
        pass
    elif args.mode == "G":
        configs = json.load(open(args.generate_config, "r"))
        generate(args, configs)
    elif args.mode == "E":
        configs = json.load(open(args.generate_config, "r"))
        extract(args, configs)
    else:
        print("no such mode %s".format(args.mode))

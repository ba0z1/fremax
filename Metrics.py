import os
import json, jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import steganalysis.steganalysis_utils as steganalysis
import evaluate
import numpy as np
from tqdm import tqdm
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def calulate_ppl_and_entropy(model, tokenizer, texts, random_start_token_num=1):
	'''
	 without context
	'''
	ppls = []
	entropys = []
	device = torch.device("cuda")
	model.to(device)
	model.eval()
	with torch.no_grad():
		for text in tqdm(texts):
			ppl = 0
			entropy = 0
			try:
				input_ids = tokenizer.encode(tokenizer.bos_token + text, return_tensors="pt").to(device)
				logits = model(input_ids).logits[0, :, :]
				probs = logits.softmax(dim=-1)

				for i in range(random_start_token_num, len(input_ids[0].tolist())-1):
					ppl -= torch.log(probs[i-1, input_ids[0, i]])
					entropy -= (probs[i-1, :] * torch.log(probs[i-1,:])).sum()
				ppls.append((ppl/len(tokenizer.encode(text))).exp().item())
				entropys.append(entropy.item())
			except:
				ppls.append(ppl)
				entropys.append(entropy)
			# ppl = 0
			# entropy = 0
			# input_ids = tokenizer.encode(tokenizer.bos_token + text, return_tensors="pt", truncation=True, max_length=1024).to(device)
			# # print(input_ids)
			# logits = model(input_ids).logits[0, :, :]
			# probs = logits.softmax(dim=-1)
			# # print(probs)
			# for i in range(random_start_token_num, len(input_ids[0].tolist())-1):
			#     ppl -= torch.log(probs[i-1, input_ids[0, i]])
			#     # print(torch.log(probs[i-1,:]))
			#     entropy -= (probs[i-1, :] * torch.log(probs[i-1,:])).sum()
			# ppls.append((ppl/len(tokenizer.encode(text))).exp().item())
			# entropys.append(entropy.item())
	return ppls, entropys

def calulate_tokens_num(model, tokenizer, texts, random_start_token_num=0):
	'''
	 without context
	'''
	# ppls = []
	# entropys = []
	device = torch.device("cpu")
	model.to(device)
	model.eval()
	tokens = []
	with torch.no_grad():

		for text in tqdm(texts):


			input_ids = tokenizer.encode(tokenizer.bos_token + text, return_tensors="pt").to(device)
			# print(len(input_ids))
			tokens.append(len(input_ids[0]))
			# logits = model(input_ids).logits[0, :, :]
			# probs = logits.softmax(dim=-1)
			# for i in range(random_start_token_num, len(input_ids[0].tolist())-1):
			#     ppl -= torch.log(probs[i-1, input_ids[0, i]])
			#     entropy -= (probs[i-1, :] * torch.log(probs[i-1,:])).sum()
			# ppls.append((ppl/len(tokenizer.encode(text))).exp().item())
			# entropys.append(entropy.item())
	return tokens

def calulate_bpw_and_payload_and_bpt(texts, bits,  tokens=None, random_start_token_num=1):
	'''
	without context
	'''
	payloads = []
	words_num = 0
	bits_num = 0
	tokens_num = 0
	for i in range(len(texts)):
		text = texts[i]
		bit = bits[i]
		bit = "".join(bit)
		payloads.append(len(bit))
		words = text.strip().split(" ")
		words_num += len(words)
		bits_num += len(bit)
		if tokens is None:
			pass
		else:
			token = tokens[i]
			tokens_num += len(token[random_start_token_num:])
	bpw = bits_num/words_num
	if token is None:
		bpt = None
	else:
		bpt = bits_num/tokens_num
	return bpw, payloads, bpt

def read_test_jsonl(stego_jsonlfile_path):
	texts = []
	bits = []
	tokens = []
	with open(stego_jsonlfile_path, "r") as f:
		for items in jsonlines.Reader(f):
			texts.append(items["stego"].strip())
			bits.append(items["bits"])
			tokens.append(items["tokens"])
	return texts, bits, tokens

def read_test_json(stego_jsonlfile_path):
	texts = []
	bits = []
	tokens = []
	with open(stego_jsonlfile_path, "r",encoding = 'utf') as f:
		if "json" in  stego_jsonlfile_path:
			texts = json.load(f)
			# for item in tmp_texts:
				# if json.loads(items)["output"] != "":
				# texts.append(item["predict"].strip())
				# bits.append(items["bits"])
				# tokens.append(items["tokens"])
		else:
			texts = f.read().split("\n")
		# texts = json.load(f)
	# print(texts[0])
	return texts, bits, tokens

def prepare_for_steganalysis_yjs(stego_jsonlfile_path, cover_corpusfile_path, output_dir,  max_num=10000, cover_file_is_json=False):
	if cover_file_is_json:
		covers_full = json.load(open(cover_corpusfile_path, "r"))
	else:
		covers_full = open(cover_corpusfile_path, "r").read().split("\n")
	covers_full = [cover for cover in covers_full if cover.strip() != ""]
	stegos_full = []
	with open(stego_jsonlfile_path, "r") as f:
		for items in jsonlines.Reader(f):
			stego = items["stego"]
			if stego.strip() == "":
				continue
			else:
				stegos_full.append(stego)

	random.shuffle(covers_full)
	random.shuffle(stegos_full)
	output_items = []
	for cover in covers_full[:max_num]:
		output_items.append({"text": cover, "label": 0})
	for stego in stegos_full[:max_num]:
		output_items.append({"text": stego, "label": 1})
	random.shuffle(output_items)
	os.makedirs(output_dir, exist_ok=True)
	json.dump(output_items, open(os.path.join(output_dir, "full.json"), "w"), indent=4)
	# json.dump(stegos_full[:max_num], open(os.path.join(out_dir, "stegos.json"), "w"))
	# json.dump(covers_full[:max_num], open(os.path.join(out_dir, "covers.json"), "w"))

def prepare_for_steganalysis(stego_jsonlfile_path, cover_corpusfile_path, output_dir,  max_num=10000):
	covers_full = open(cover_corpusfile_path, "r").read().split("\n")
	covers_full = [cover for cover in covers_full if cover.strip() != ""]
	stegos_full = []
	with open(stego_jsonlfile_path, "r") as f:
		for items in jsonlines.Reader(f):
			stego = items["stego"]
			if stego.strip() == "":
				continue
			else:
				stegos_full.append(stego)
	#
	# data = json.load(open(stego_jsonlfile_path, "r"))
	#
	# for item in data:
	#     stegos_full.append(item["predict"].strip())
	# with open(stego_jsonlfile_path, "r") as f:
	#     for items in jsonlines.Reader(f):
	#         stego = items["stego"]
	#         if stego.strip() == "":
	#             continue
	#         else:
	#             stegos_full.append(stego)

	random.shuffle(covers_full)
	random.shuffle(stegos_full)
	output_items = []
	for cover in covers_full[:max_num]:
		output_items.append({"text": cover, "label": 0})
	for stego in stegos_full[:max_num]:
		output_items.append({"text": stego, "label": 1})
	random.shuffle(output_items)
	os.makedirs(output_dir, exist_ok=True)
	json.dump(output_items, open(os.path.join(output_dir, "full.json"), "w"), indent=4)
	# json.dump(stegos_full[:max_num], open(os.path.join(out_dir, "stegos.json"), "w"))
	# json.dump(covers_full[:max_num], open(os.path.join(out_dir, "covers.json"), "w"))


def classifier(data_dir, model_type, output_dir, pretrained_model_name_or_path=None, do_train=True, do_test=True):
	model_path = os.path.join(output_dir, model_type)
	os.makedirs(model_path, exist_ok=True)
	if pretrained_model_name_or_path is None:
		try:
			tokenizer = AutoTokenizer.from_pretrained(output_dir)
		except:
			tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	else:
		tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	use_plm = pretrained_model_name_or_path is not None
	model = steganalysis.load_model(model_type, tokenizer, use_plm)

	if do_train:
		data_filepath = os.path.join(data_dir, "full.json")
		train_data, val_data, test_data = steganalysis.load_data(data_filepath, tokenizer, cutoff_len=128, val_ratio=0.1, test_ratio=0.1, do_train=do_train)
	else:
		data_filepath = os.path.join(data_dir, "test.json")
		train_data, val_data, test_data = steganalysis.load_data(data_filepath, tokenizer, cutoff_len=128,
																 val_ratio=0.1, test_ratio=0.1, do_train=do_train)

	if use_plm:
		training_args = {"batch_size": 16,
						 "per_device_train_batch_size": 16,
						 "num_epochs": 10,
						 "learning_rate": 1e-4,
						 "save_steps": 1000,
						 "eval_steps": 1000,
						 "warmup_steps": 10,
						 "logging_steps": 100}
	else:
		training_args = {"batch_size": 160,
						 "per_device_train_batch_size": 160,
						 "num_epochs": 10,
						 "learning_rate": 1e-3,
						 "save_steps": 100,
						 "eval_steps": 100,
						 "warmup_steps": 10,
						 "logging_steps": 10}
	if do_train:
		steganalysis.train(model, tokenizer, output_dir,
						   train_data, val_data, test_data,
						   training_args,
						   resume_from_checkpoint=False)

	if do_test and os.path.exists(model_path):
		model_to_predict = steganalysis.load_model(model_type, tokenizer, use_plm, checkpoint=os.path.join(output_dir))
		steganalysis.test(model_to_predict, tokenizer, test_data, output_dir)


def load_huggingface_metrics(metric_name):
	metric = evaluate.load(os.path.join("/home/sanxinshidai/yjs/huggingface_metrics/metrics", metric_name))
	return metric


def calulate_bleu(predictions, references, tokenizer=None):
	from nmt import bleu
	if tokenizer is None:
		from nmt import tokenizer_13a
		tokenizer = tokenizer_13a.Tokenizer13a()

	if isinstance(references[0], str):
		references = [[ref] for ref in references]

	references = [[tokenizer(r) for r in ref] for ref in references]
	predictions = [tokenizer(p) for p in predictions]
	score = bleu.compute_bleu(
		reference_corpus=references, translation_corpus=predictions, max_order=4, smooth=False
	)
	(bleu, precisions, bp, ratio, translation_length, reference_length) = score
	cum_bleu = precisions[0] * 0.25 + precisions[1] *0.25 + precisions[2] * 0.25 + precisions[3] * 0.25

	return {
		"Cumulative BLEU-4": cum_bleu
	}

def calculate_rouge(predictions, references, tokenizer=None):
	from rouge_score import rouge_scorer, scoring
	if tokenizer is None:
		from nmt import tokenizer_13a
		tokenizer = tokenizer_13a.Tokenizer13a()

	rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
	multi_ref = isinstance(references[0], list)

	class Tokenizer:
		"""Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

		def __init__(self, tokenizer_func):
			self.tokenizer_func = tokenizer_func

		def tokenize(self, text):
			return self.tokenizer_func(text)

	if tokenizer is not None:
		tokenizer = Tokenizer(tokenizer)

	use_aggregator = False
	scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)

	if use_aggregator:
		aggregator = scoring.BootstrapAggregator()
	else:
		scores = []

	for ref, pred in zip(references, predictions):
		if multi_ref:
			score = scorer.score_multi(ref, pred)
		else:
			score = scorer.score(ref, pred)
		if use_aggregator:
			aggregator.add_scores(score)
		else:
			scores.append(score)

	if use_aggregator:
		result = aggregator.aggregate()
		for key in result:
			result[key] = result[key].mid.fmeasure

	else:
		result = {}
		for key in scores[0]:
			result[key] = list(score[key].fmeasure for score in scores)

	return result


def calulate_bleu_and_rouge(predictions, references, tokenizer=None):
	bleu_results = calulate_bleu(predictions, references, tokenizer)
	rouge_results = calculate_rouge(predictions, references, tokenizer)
	results = {'Cumulative BLEU-4':bleu_results['Cumulative BLEU-4']}
	for k in rouge_results:
		results[k] = np.mean(rouge_results[k])
	for k,v in results.items():
		print(k,v)
	return  results


def distint_n(tokenized_text):
	counter_2 = Counter()
	total_2 = 0
	distinct_2 = 0
	distinct_2, total_2, counter_2 = distinct_n(
		tokenized_text, 2, distinct_2, total_2, counter_2)  # Need to set n
	tmp_distinct_2 = distinct_2 / total_2

	# 3_Distinct
	counter_3 = Counter()
	total_3 = 0
	distinct_3 = 0
	distinct_3, total_3, counter_3 = distinct_n(
		tokenized_text, 3, distinct_3, total_3, counter_3)  # Need to set n
	tmp_distinct_3 = distinct_3 / total_3

	# 4_Distinct
	counter_4 = Counter()
	total_4 = 0
	distinct_4 = 0
	distinct_4, total_4, counter_4 = distinct_n(
		tokenized_text, 4, distinct_4, total_4, counter_4)
	# Need to set n
	tmp_distinct_4 = distinct_4 / total_4
	print("distinct-2", tmp_distinct_2)
	print("distinct-3", tmp_distinct_3)
	print("distinct-8", tmp_distinct_4)

	return tmp_distinct_3


# def distinct_n(example, n, n_distinct, n_total, counter):
#     """
#     Gives the number of distinct n-grams as well as the total n-grams
#     Args:
#         example: input text
#         n: n-grams size (i.e., the n)
#         n_distinct: distinct n-grams in previous iteration
#         n_total: total n-grams in previous iteration
#         counter: token counter in previous iteration, i.e., how many times a token appeared
#
#     """
#     for token in zip(*(example[i:] for i in range(n))):
#         if token not in counter:
#             n_distinct += 1
#         elif counter[token] == 1:
#             n_distinct -= 1
#         counter[token] += 1
#         n_total += 1
#     if n_total == 0:
#         n_total = 1
#     return n_distinct, n_total, counter


#
def distinct_n(tokens, n):
	# 将 tokens 列表中的元素按照长度为 n 的窗口进行切片
	ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
	distinct_ngrams = set(ngrams)
	return len(distinct_ngrams) / len(ngrams)

#
def calculate_distinct(data, n, tokenizer):
	# 遍历 data 列表的每个元素，将其进行分词，并将分词结果添加到 all_tokens 列表中
	all_tokens = []
	for entry in tqdm(data):
		tokens = tokenizer.tokenize(entry)
		all_tokens.extend(tokens)
	distinct = distinct_n(all_tokens, n)
	return distinct


if __name__ == '__main__':
	model_name_or_path = "gpt2-large"
	# model_name_or_path = "bert-base-uncased"
	cache_dir = "/data3/ADG_P/huggingface/hub"
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
	# for t in [1.13828125 + i*2**(-9) for i in range(0,32)]:
	# for p in [1.0]:
		# for theta in[0.1,0.2,0.3,0.5,0.08,0.11,0.12,0.13,0.14,0.15,0.105,0.115,0.125,0.135,0.145,1,-0.1]:

		# for t in [0.5]:
	for beta in [ 2.0]:
		for alpha in [ 
		# "/data3/ADG_P/gpt2_movies/3eps/movie_gpt2_3eps-ac-10000-512-precision52.jsonl",
		# "/data3/ADG_P/gpt2_movies/3eps/movie_gpt2_3eps-plain-10000-512-temperature1.0-topp1.0-topk100000.jsonl",
		# "/data3/ADG_P/gpt2_movies/30eps/movie_gpt2_30eps-ac-10000-512-precision52.jsonl",
		# "/data3/ADG_P/gpt2_movies/30eps/movie_gpt2_30eps-plain-10000-512-temperature1.0-topp1.0-topk100000.jsonl"
		# "/data3/ADG_P/corpus/movie2020.txt"
		# "/data3/ADG_P/redditlm/softmax_with_frequency/movie/30ep/alpha=0.5_beta=2.0.jsonl",
		#  2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0
		0
		]:
			# stego_jsonlfile_path = f"/data3/ADG_P/redditlm/softmax_with_frequency/test_baseline/alpha={alpha}.jsonl"
			# /data3/ADG_P/RedditLM/algorithms/models_trained-plain-1000-512-T=1.0-P=1.0.jsonl
			# stego_jsonlfile_path = f"/data3/ADG_P/redditlm/softmax_with_frequency/alpha_and_beta/alpha={alpha}_beta={beta}.jsonl"
			stego_jsonlfile_path = f"/data3/discop/Discop/discop_movie_output.json"
			save_full_dir = "/data3/ADG_P/redditlm/softmax_with_frequency/theta/metric_results"
			# prepare_for_steganalysis(stego_jsonlfile_path, "/data3/ADG_P/tweet2020.txt", save_full_dir, max_num=1000)

			# stego_jsonlfile_path ="predictions_0616_ADG_P=0.966.json"
			# save_full_dir="/data3/ADG_P/p=0.966"

			result_file_path = stego_jsonlfile_path.split(".")[0] +"_result.json"
			# prepare_for_steganalysis(stego_jsonlfile_path, "/data3/ADG_P/tweet2020.txt", save_full_dir, max_num=1000)

			texts, bits, tokens = read_test_json(stego_jsonlfile_path)
			
			texts = texts[0:1000]
			bit_num = 0
			for i in bits:
				for j in i:
					bit_num += len(j)
			# print(bit_num)

			# 計算rough
			#
			#
			ppls, entropys = calulate_ppl_and_entropy(model, tokenizer, texts)
			# print(np.mean(ppls), np.mean(entropys))
			tokens = calulate_tokens_num(model, tokenizer, texts)
			# print(np.mean(tokens))
			distinct_2 = calculate_distinct(texts, 2, tokenizer)
			distinct_3 = calculate_distinct(texts, 3, tokenizer)
			distinct_4 = calculate_distinct(texts, 4, tokenizer)
			# print(f'distinct2='.format(np.mean(distinct_2)))
			# print(f'distinct3='.format(np.mean(distinct_3)))
			# print(f'distinct4='.format(np.mean(distinct_4)))


			# evaluations = []
			# for i in range(len(texts)):
			#     evaluation = {}
			#     evaluation['sentence'] = texts[i]
			#     evaluation['ppl'] = ppls[i]
			#     evaluation['entropy'] = entropys[i]
			#     evaluation['tokens_num'] = tokens[i]
			#
			#     evaluations.append(evaluation)
			# with open(result_file_path,"w+",encoding = 'utf-8') as f:
			#     json.dump(evaluations,f)


			# bpw, payloads, bpt = calulate_bpw_and_payload_and_bpt(texts, bits, tokens)
			# print(np.mean(ppls), np.mean(entropys), bpw, np.mean(payloads), bpt)
			# print(np.mean(ppls), np.mean(entropys))
			# one_result={"ppls_mean": np.mean(ppls), "ppls_std": np.std(ppls),"ppls_med": np.median(ppls),
			# 			"entropys_mean": np.mean(entropys),"entropys_std": np.std(entropys),"entropys_med": np.median(entropys),
			# 			"tokens_nums_mean": np.mean(tokens),"tokens_nums_std": np.std(tokens),"tokens_nums_med": np.median(tokens),
			# 			"dist2":distinct_2,"dist3":distinct_3,"dist4":distinct_4, "bpt": bit_num/np.sum(tokens)}
			# print("ppls_mean\tppls_std\tppls_med\tentropys_mean\tentropys_std\tentropys_med\ttokens_nums_mean\ttokens_nums_std\ttokens_nums_med\tdist2\tdist3\tdist4\tbpt\t")
			print(alpha)
			# print(f'{round(bit_num/np.sum(tokens),2)}\t{round(bit_num/len(tokens),2)}')
			print(f"{round(np.mean(ppls),2)}\t{round(np.std(ppls),2)}\t{round(np.median(ppls),2)}\t{round(np.mean(entropys),2)}\t{round(np.std(entropys),2)}\t{round(np.median(entropys),2)}\t{round(np.mean(tokens),2)}\t{round(np.std(tokens),2)}\t{round(np.median(tokens),2)}\t{round(distinct_2,2)}\t{round(distinct_3,2)}\t{round(distinct_4,2)}\t{round(bit_num/np.sum(tokens),2)}\t{round(bit_num/len(tokens),2)}")

			# json.dump(one_result,open('/data3/ADG_P/redditlm/softmax_with_frequency/test'+f'result_alpha={alpha}.json', "w+"))
			data_dir=save_full_dir
			# for model_type in ["cnn","rnn"]:
			#
			#     output_dir = os.path.join("/data3/ADG_P/SteganalysisData/models-10000sample_no_BERT_GridSearch0726", "__".join([model_type] + data_dir.split("/")))
			#
			#
			#     # classifier(data_dir, model_type, output_dir, pretrained_model_name_or_path="bert-base-uncased")
			#     classifier(data_dir, model_type, output_dir)

	# for sub_data in ["adgandhc", ]:
#     for model_type in ["cnn", "rnn", "fcn"]:
#         data_dir = "/data/yjs/StegoData/RedditGenshin_withoutcontext/"+sub_data
#         output_dir = os.path.join("/data/yjs/StegoData/models", "__".join([model_type] + data_dir.split("/")))
#         classifier(data_dir, model_type, output_dir, cutoff_len=256, padding=True)
	# for sub_data in ["ac", "hc5", "adg", "adgv2", "adgandhc", "AI"]:
	#     data_dir = "/data/yjs/StegoData/tweet/"+sub_data
	#     model_type = "cnn"
	#     output_dir = os.path.join("/data/yjs/StegoData/models", "__".join([model_type] + data_dir.split("/")))
	#     classifier(data_dir, model_type, output_dir)

	# for sub_data in ["AI"]:
	#     data_dir = "/data/yjs/StegoData/tweet/"+sub_data
	#     model_type = "cnn"
	#     pretrained_model_name_or_path = "bert-base-uncased"
	#     output_dir = os.path.join("/data/yjs/StegoData/models", "__".join([model_type, pretrained_model_name_or_path] + data_dir.split("/")))
	#     classifier(data_dir, model_type, output_dir, pretrained_model_name_or_path=pretrained_model_name_or_path, do_train=False)


   # for sub_data in ["AI"]:
   #      data_dir = "/data/yjs/StegoData/tweet/"+sub_data
   #      model_type = "fcn"
   #      pretrained_model_name_or_path = "bert-base-uncased"
   #      output_dir = os.path.join("/data/yjs/StegoData/models", "__".join([model_type] + data_dir.split("/")))
   #      classifier(data_dir, model_type, output_dir, )

   # calulate_bleu_and_rouge(["中国歼-20四代机曾在试飞时携带一枚新型的空空导弹亮相｡这枚导弹安置在歼-20四代机的侧弹舱位置,在整个试飞过程中一直处于伸出舱外准备攻击的姿态｡",
   #                          "中国歼-20四代机曾在试飞时携带一枚新型的空空导弹亮相｡这枚"],
   #
   #                         ["中国歼-20四代机曾在试飞时携带一枚新型的空空导弹亮相｡这枚导弹安置在歼-20四代机的侧弹舱位置,在整个试飞过程中一直处于伸出舱外准备攻击的姿态｡",
   #                          "中国歼-20四代机曾在试飞时携带一枚新型的空空导弹亮相｡这枚"
   #                          ])


import os
import json
import random
import csv
import torch
import numpy as np
import math
import transformers

from steganalysis.models import birnn as BiRNN, cnn as CNN, lstmatt as LSTMATT, fcn as FCN, r_bilstm_c as RBC,\
    bilstm_dense as BLSTMDENSE,sesy as SESY, gnn as GNN

from sklearn.model_selection import train_test_split
from datasets import load_dataset
import evaluate
#
# task_metrics = {"steganalysis" : "accuracy",
#                 "graph_steganalysis" : "accuracy", }
# time_stamp = "-".join(time.ctime().split())


def split_train_dev_test(data_dir, split_ratio):
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "covers.json"), 'r', encoding='utf-8') as f:
        covers = json.load(f)
    covers = list(filter(lambda x: x not in ['', None], covers))
    random.shuffle(covers)
    with open(os.path.join(data_dir, "stegos.json"), 'r', encoding='utf-8') as f:
        stegos = json.load(f)
    stegos = list(filter(lambda x: x not in ['', None], stegos))
    random.shuffle(stegos)
    texts = covers + stegos
    labels = [0] * len(covers) + [1] * len(stegos)
    val_ratio = (1 - split_ratio) / split_ratio
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,
                                                                          train_size=split_ratio)
    train_texts, val_texts, train_labels, val_labels, = train_test_split(train_texts, train_labels,
                                                                         train_size=1 - val_ratio)
    def write2file(X, Y, filename):
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for x, y in zip(X, Y):
                writer.writerow([x, y])

    write2file(train_texts, train_labels, os.path.join(data_dir, "train.csv"))
    write2file(val_texts, val_labels, os.path.join(data_dir, "val.csv"))
    write2file(test_texts, test_labels, os.path.join(data_dir, "test.csv"))


def load_model(model_type, tokenizer, use_plm=False, checkpoint=None):
    # set model
    Model_configs_all = json.load(open("steganalysis/steganalysis_default.json", "r"))
    VOCAB_SIZE = tokenizer.vocab_size
    class_num = Model_configs_all["class_num"]
    if not use_plm:
        if model_type.lower() in ["ts-csw", "cnn"]:
            model_configs = Model_configs_all["CNN"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = CNN.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["birnn", "rnn"]:
            model_configs = Model_configs_all["RNN"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = BiRNN.TC(**{**model_configs, "class_num": class_num})
        elif model_type.lower() in ["fcn", "fc"]:
            model_configs = Model_configs_all["FCN"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = FCN.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["lstmatt", "att"]:
            model_configs = Model_configs_all["LSTMATT"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = LSTMATT.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["r-bilstm-c", "r-b-c", "rbc", "rbilstmc"]:
            model_configs =  Model_configs_all["RBiLSTMC"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = RBC.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["bilstmdense", "bilstm-dense", "bilstm_dense", "bi-lstm-dense"]:
            model_configs =  Model_configs_all["BiLSTMDENSE"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = BLSTMDENSE.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["sesy"]:
            model_configs =  Model_configs_all["SESY"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = SESY.TC(**{**model_configs, "class_num": class_num, })
        elif model_type.lower() in ["gnn"]:
            model_configs =  Model_configs_all["GNN"]
            model_configs["vocab_size"] = VOCAB_SIZE
            model = GNN.TC(**{**model_configs, "class_num": class_num, })
        else:
            print("no such model, exit")
            exit()
        if checkpoint is not None:
            print("---------------------loading model from {}------------".format(checkpoint))
            model = torch.load(os.path.join(checkpoint, "model_and_state.bin"))
    else:
        if checkpoint is not None:
            print("---------------------loading model from {}------------".format(checkpoint))
            model_name_or_path = checkpoint
        else:
            print("-------------loading pretrained language model from huggingface--------------")
            model_name_or_path = Model_configs_all["pretrained_model_name_or_path"]

        if model_type.lower() in ["ts-csw", "cnn"]:
            model_configs = Model_configs_all["CNN"]
            model = CNN.BERT_TC.from_pretrained(model_name_or_path,
                                                **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["birnn", "rnn"]:
            model_configs = Model_configs_all["RNN"]
            model = BiRNN.BERT_TC.from_pretrained(model_name_or_path,
                                                **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["fcn", "fc"]:
            model_configs = Model_configs_all["FCN"]
            model = FCN.BERT_TC.from_pretrained(model_name_or_path, **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["lstmatt", "att"]:
            model_configs = Model_configs_all["LSTMATT"]
            model = LSTMATT.BERT_TC.from_pretrained(model_name_or_path,
                                                **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["r-bilstm-c", "r-b-c", "rbc", "rbilstmc"]:
            model_configs = Model_configs_all["RBiLSTMC"]
            model = RBC.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["bilstmdense", "bilstm-dense", "bilstm_dense", "bi-lstm-dense"]:
            model_configs = Model_configs_all["BiLSTMDENSE"]
            model = BLSTMDENSE.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{** model_configs, "class_num": class_num, })
        elif model_type.lower() in ["sesy"]:
            model_configs = Model_configs_all["SESY"]
            model = SESY.BERT_TC.from_pretrained(model_name_or_path,
                                                    **{** model_configs, "class_num": class_num, })
        else:
            print("no such model, exit")
            exit()


    print("Model Configs")
    print(json.dumps({**{"MODEL_TYPE": model_type}, **model_configs, }))
    return model


def load_data(data_filepath, tokenizer, cutoff_len=128, val_ratio=0.1, test_ratio=0.1, do_train=True, padding=False):
    if not do_train:
        train_data = None
        val_data = None
        test_data = load_dataset(path="json", data_files=data_filepath)["train"]
        return train_data, val_data, test_data

    def prepare_for_training(data_point):
        input_text = data_point["text"]
        label = data_point["label"]
        result = tokenizer(
            input_text,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length" if padding else False,
            return_tensors=None,
        )

        result["labels"] = int(label)

        return result

    data = load_dataset(path="json", data_files=data_filepath)
    test_set_size = round(test_ratio * len(data["train"]))
    val_set_size = round(val_ratio * len(data["train"]))

    if test_set_size > 0:
        train_test = data["train"].train_test_split(test_size=test_set_size, shuffle=True, seed=42)
        train_data = train_test["train"].shuffle()
        test_data = train_test["test"].shuffle().map(prepare_for_training)
    else:
        train_data = data["train"].shuffle()
        test_data = None
    train_val = train_data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(prepare_for_training)
    val_data = train_val["test"].shuffle().map(prepare_for_training)

    data_dir = os.path.split(data_filepath)[0]
    train_data.to_json(path_or_buf=os.path.join(data_dir, "train.json"))
    val_data.to_json(path_or_buf=os.path.join(data_dir, "val.json"))
    test_data.to_json(path_or_buf=os.path.join(data_dir, "test.json"))
    # train_data = load_dataset(path="json", data_files=os.path.join(data_dir,"train.jsonl"),
    #                           num_proc=8,)["train"].shuffle().map(generate_and_tokenize_prompt)
    # val_data = load_dataset(path="json", data_files=os.path.join(data_dir, "valid.jsonl"),
    #                         num_proc=8,)["train"].shuffle().map(generate_and_tokenize_prompt)
    # test_data = load_dataset(path="json", data_files=os.path.join(data_dir, "test.jsonl"),
    #                          num_proc=8,)["train"].shuffle().map(generate_and_tokenize_prompt)

    return train_data, val_data, test_data



def train(model, tokenizer, output_dir, train_data, val_data, test_data, training_args=None, resume_from_checkpoint=False):
    gradient_accumulation_steps = training_args['batch_size'] // training_args[
        'per_device_train_batch_size'] if "gradient_accumulation_steps" not in training_args else training_args[
        'gradient_accumulation_steps']
    metric = evaluate.load("/data3/ADG_P/huggingface_metrics/metrics/accuracy/accuracy.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0], axis=1)
        return metric.compute(predictions=predictions, references=labels)


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=training_args['per_device_train_batch_size'],
            per_device_eval_batch_size=training_args['per_device_train_batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=training_args['warmup_steps'],
            num_train_epochs=training_args['num_epochs'],
            learning_rate=training_args['learning_rate'],
            logging_steps=training_args['logging_steps'],
            evaluation_strategy="steps" if len(val_data) > 0 else "no",
            save_strategy="steps",
            eval_steps=training_args["eval_steps"] if len(val_data) > 0 else None,
            save_steps=training_args["save_steps"],
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        ),
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer
        ),
    )
    print("trainer.train")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Save checkpointing...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if not isinstance(model, transformers.PreTrainedModel):
        torch.save(trainer.model, os.path.join(output_dir, "model_and_state.bin"))

    if test_data is not None:
        test(trainer.model, tokenizer, test_data, output_dir)


def test(model, tokenizer, test_data, output_dir):
    metric = evaluate.load("/data3/ADG_P/huggingface_metrics/metrics/accuracy/accuracy.py")
    metric2= evaluate.load("/data3/ADG_P/huggingface_metrics/metrics/f1/f1.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0], axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = transformers.Trainer(
        model=model,
        compute_metrics=compute_metrics,
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer
        ),
    )
    with torch.no_grad():
        results = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test_")
    for k, v in results.items():
        print(k, v)
    json.dump(results, open(os.path.join(output_dir, "results.json", ), "w"), indent=4)

import os
import sys
import pathlib

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
cwd = os.getcwd()

sys.path.append(cwd)
from transformers import AutoTokenizer
from eyetrackpy.data_generator.fixations_predictor.models.generate_fixations_predictor import (
    CreateFixationsPredictorModel,
)
from transformers import DataCollatorWithPadding
import torch
import pandas as pd

if __name__ == "__main__":
    sentences = ["Hello, how are you?"]
    #--------------------------------
    #example without mapping to a model
    create_fixations = CreateFixationsPredictorModel(version=1)
    fixations= create_fixations.predict_fromtext(
        sentences = sentences
    )
    print(fixations)
    #--------------------------------
    #example mapping to a model
    model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    # Tokenize the text
    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    # Create a simple dataset from the tokenized inputs
    dataset = {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }
    dataloader = torch.utils.data.DataLoader(
        [dataset], batch_size=batch_size, collate_fn=collator
    )
    create_fixations = CreateFixationsPredictorModel(model_name, tokenizer=tokenizer, version=2)
    fixations_all = create_fixations.process_data(
        dataloader
    )
    print(fixations_all)
    #--------------------------------
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
from rlhf_rw.reward_utils.dataset_proceser_reward import DatasetProceserReward
import torch
from eyetrackpy.data_generator.fixations_predictor.models.generate_fixations_predictor import (
    CreateFixationsPredictorModel,
)
from transformers import (
    DataCollatorWithPadding,
)
from utils.utils import load_tokenizer

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size = 2

    tokenizer = load_tokenizer(model_name)
    dataset_name = "OpenAssistant/oasst1"
    split = "train[:2%]"
    dataset_procesor = DatasetProceserReward.from_datasets(
        dataset_name=dataset_name,
        model_name=model_name,
        split=split,
        tokenizer=tokenizer,
    )
    raw_datasets = dataset_procesor.preprocess_data_reward(
        tokenizer=tokenizer, chosen_name="chosen_chat", rejected_name="rejected_chat"
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    text_test = train_dataset["chosen_chat"]

    train_dataset = train_dataset.rename_column("input_ids_chosen", "input_ids")
    train_dataset = train_dataset.rename_column(
        "attention_mask_chosen", "attention_mask"
    )

    # create new dataset with the correct columns
    train_dataset.set_format(columns=["input_ids", "attention_mask"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    # in dataset you can pass the HF dataset with the correct columns or the tokens list like in preovious example
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=collator
    )

    # create_fixations = CreateFixationsPredictorForModel(model_name, tokenizer=tokenizer)
    create_fixations = CreateFixationsPredictorModel(model_name, tokenizer=tokenizer)
    model_tok_combined_all, fix_tok_combined_all = create_fixations.process_data(
        dataloader
    )
    print("angela")

import os
import sys

# sys.path.append("../..")
# path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.append(path)
# path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
# sys.path.append(path)
# path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
# sys.path.append(path)
cwd = os.getcwd()

sys.path.append(cwd)
from rlhf_rw.reward_utils.dataset_proceser_reward import DatasetProceserReward
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner

from transformers import AutoTokenizer


model_name = "t5-small"
tokenizer_fix = AutoTokenizer.from_pretrained(
    model_name, cache_dir="./cache/models", model_max_length=2048
)
split = "train[:80%]"
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer_model = AutoTokenizer.from_pretrained(model_name)
tokenizer_model.add_special_tokens({"pad_token": "[PAD]"})
print(tokenizer_model.all_special_tokens)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer_model = AutoTokenizer.from_pretrained(model_name)
tokenizer_model.add_special_tokens({"pad_token": "[PAD]"})
tokenizer_model.add_special_tokens(
    {"additional_special_tokens": ["<|start_header_id|>", "<|end_header_id|>"]}
)
print(tokenizer_model.all_special_tokens)


# dataset_name = "timdettmers/openassistant-guanaco"
# data = load_dataset(dataset_name, split=split)
# text_test = train_dataset["text"][90]

dataset_name = "OpenAssistant/oasst1"
dataset_procesor = DatasetProceserReward.from_datasets(
    dataset_name=dataset_name,
    model_name=model_name,
    split=split,
    tokenizer=tokenizer_model,
)
raw_datasets = dataset_procesor.preprocess_data_reward(
    tokenizer=tokenizer_model, chosen_name="chosen_chat", rejected_name="rejected_chat"
)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
text_test = train_dataset["chosen_chat"]

# text_test = re.sub(r"[\'`]" , '', text_test)
# text_test = ["hello I am desperated"]


def tokenize_text_both_tokenizers(
    tokenizer_model: AutoTokenizer,
    tokenizer_fix: AutoTokenizer,
    text_originals=None,
    text_tokenized_first_orignal=None,
):
    if text_originals is None and text_tokenized_first_orignal is None:
        raise ValueError(
            "text_originals or text_tokenized_first_orignal must be provided"
        )
    if text_originals is not None and text_tokenized_first_orignal is not None:
        raise ValueError(
            "text_originals and text_tokenized_first_orignal cannot be provided at the same time"
        )
    if text_originals is not None:
        if not isinstance(text_originals, list):
            text_originals = [text_originals]
        text_tokenized_first_orignal = tokenizer_model(
            text_originals, padding=True, truncation=True, add_special_tokens=False
        )
    texts = []
    for i in range(len(text_tokenized_first_orignal["input_ids"])):
        texts.append(
            tokenizer_model.decode(
                text_tokenized_first_orignal["input_ids"][i], skip_special_tokens=True
            )
        )
    if len(texts) == 1:
        texts = texts[0]
    # texts = texts[0]
    text_tokenized_first = tokenizer_model(
        texts, padding=True, truncation=True, add_special_tokens=True
    )
    text_tokenized_second = tokenizer_fix(
        texts, padding=True, truncation=True, add_special_tokens=True
    )
    return texts, text_tokenized_first, text_tokenized_second


for num, text_test in enumerate(train_dataset["chosen_chat"]):
    try:
        texts, text_tokenized_model, text_tokenized_fix = tokenize_text_both_tokenizers(
            tokenizer_model, tokenizer_fix, text_originals=text_test
        )
        # --------------------------------------------------------------------------------------------
        tokens_idx_mapped, tokens_id_mapped, words_str_mapped, tokens_str_mapped = (
            TokenizerAligner().align_tokens(texts, text_tokenized_model, text_tokenized_fix, return_all=True)
        )
        # for i in range(len(tokens_idx_mapped)):
        #     print(i, words_str_mapped[i], "-----", tokens_str_mapped[i])
        print(num, "CORRECT ALIGNMENT")
    # --------------------------------------------------------------------------------------------
    except Exception as e:
        print(num, ":", e)
        word_token_idx_model = text_tokenized_model.word_ids()
        words_model = TokenizerAligner.text_to_words(
            texts, text_tokenized_model, word_token_idx_model
        )
        word_tokens_model = TokenizerAligner().words_tokens_to_str(
            word_token_idx_model, text_tokenized_model
        )
        len(word_tokens_model)
        for i in range(len(words_model)):
            print(i, words_model[i], "-----", word_tokens_model[i])
        print(num, "INCORRECT ALIGNMENT")

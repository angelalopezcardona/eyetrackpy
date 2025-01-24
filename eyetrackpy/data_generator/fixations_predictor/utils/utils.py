from transformers import AutoTokenizer


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_eos_token = True
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.padding_side = "right"
    chat_tokens = list(set(re.findall(r"(<.*?>)", tokenizer.default_chat_template)))
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.additional_special_tokens + chat_tokens}
    )
    return tokenizer

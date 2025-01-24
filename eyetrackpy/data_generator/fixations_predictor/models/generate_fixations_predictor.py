import os
import sys
from transformers import AutoTokenizer
import pathlib

# sys.path.append('/home/csp/repo/LLMs/eye_transformer/')
# print("CWD", os.getcwd(), "PATH", sys.path)
from collections.abc import Iterable

sys.path.append("../..")
import torch

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
)

from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import (
    FixationsPredictor_1,
)
from eyetrackpy.data_generator.fixations_predictor_trained_2.fixations_predictor_model_2 import (
    FixationsPredictor_2,
)

# from fixations_predictor_2 import FixationsPredictor_2
import pathlib

""" We use this class to predict fixations. 
The idea is that this is a simple version to text it, and some function will ve copied to other models, like the reward model """


class DeprecatedCreateFixationsPredictor:
    def __init__(
        self,
        tokenizer=None,
        version=1,
        remap=False,
        *argv,
        **karg,
    ):
        self.tokenizer = tokenizer
        self.remap = remap

    def _load_fx_model_v1(self):
        path = str(
            pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve()
        )
        self.fixTokenizer = AutoTokenizer.from_pretrained(
            "t5-small", cache_dir="./cache/models", model_max_length=2048
        )
        # self.fixTokenizer.pad_token = self.fixTokenizer.eos_token

        self.modelTokenizer = self.tokenizer
        FP_dir = os.path.join(
            path,
            "fixations_predictor_trained_1",
            "FPmodels",
            "T5-tokenizer-BiLSTM-TRT-12-concat-3",
        )
        vocab_size = 32128
        empty_emb = nn.Embedding(vocab_size, 512)

        self.FP_model = FixationsPredictor_1(
            empty_emb,
            128,
            0.2,
            self.fixTokenizer,
            self.modelTokenizer,
            remap=self.remap,
        )
        self.FP_model.load_state_dict(torch.load(FP_dir))
        self.FP_model.eval()
        for param in self.FP_model.parameters():
            param.requires_grad = False

    def process_data(self, sentences):
        fixations, fixations_atten = self.FP_model.forward(sentences)
        return fixations, fixations_atten


class DeprecatedCreateFixationsPredictorForModel(DeprecatedCreateFixationsPredictor):
    def __init__(
        self,
        model_name,
        tokenizer=None,
        version=1,
        remap=False,
        *argv,
        **karg,
    ):
        super().__init__(
            tokenizer=tokenizer, version=version, remap=remap, *argv, **karg
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            device_map="auto",
            *argv,
            **karg,
        )
        config = model.config
        self.model_name = model_name
        self.tokenizer = tokenizer
        if version == 1:
            self.load_fx_model_v1(hidden_size=config.hidden_size)

    def load_fx_model_v1(self, hidden_size):
        self._load_fx_model_v1()

        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, hidden_size)
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def process_data(self, dataloader: torch.utils.data.DataLoader):
        fixations_all = []
        for batch in dataloader:
            # print("-----------------------------------------------")
            # print(batch)
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                model_tok,
                fix_tok,
                sentences,
            ) = self.FP_model._compute_mapped_fixations_v1(batch["input_ids"])
            fixations_all.append(
                {
                    "fixations": fixations,
                    "fixations_attention_mask": fixations_attention_mask,
                    "mapped_fixations": mapped_fixations,
                    "fix_tok": fix_tok,
                    "model_tok": model_tok,
                    "sentences": sentences,
                }
            )

        return fixations_all


class CreateFixationsPredictorModel:
    def __init__(
        self,
        model_name=None,
        tokenizer=None,
        version=1,
        remap=False,
        *argv,
        **karg,
    ):
        self.model_name = model_name
        if model_name:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                device_map="auto",
                *argv,
                **karg,
            )
            config = model.config
            hidden_size = config.hidden_size
            self.modelTokenizer = tokenizer
            self.remap = remap
        else:
            # you will only compute fixations without remapping
            # if a tokenizer is passed we use it to detokenize the sentence

            self.modelTokenizer = tokenizer
            self.remap = False
            hidden_size = None

        if version not in [1, 2]:
            raise ValueError("Fixations predictor Version must be 1 or 2")
        self.version = version
        if self.version == 1:
            self.load_fx_model_v1(hidden_size)
        elif self.version == 2:
            self.load_fx_model_v2(hidden_size)

    def load_fx_model_v1(self, hidden_size):
        self.FP_model = FixationsPredictor_1(
            hidden_dim=128,
            drop_out=0.2,
            modelTokenizer=self.modelTokenizer,
            remap=self.remap,
        )
        if self.model_name:
            # this layers only need to add or concatenate to the hidden size of the model (like llama)
            self.fixations_embedding_projector = nn.Sequential(
                nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, hidden_size)
            )
            self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def load_fx_model_v2(self, hidden_size):
        self.FP_model = FixationsPredictor_2(
            modelTokenizer=self.modelTokenizer, remap=self.remap
        )
        if self.model_name:
            # this layers only need to add or concatenate to the hidden size of the model (like llama)
            self.fixations_embedding_projector = nn.Sequential(
                nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, hidden_size)
            )
            self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def _compute_mapped_fixations(self, batch_data):
        input_ids_original = batch_data
        if self.version == 1:
            if isinstance(batch_data, Iterable) and not isinstance(
                batch_data, torch.Tensor
            ):
                if "input_ids" in batch_data:
                    input_ids_original = batch_data["input_ids"]
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                model_tok,
                fix_tok,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids_original)
        if self.version == 2:
            if isinstance(batch_data, Iterable) and not isinstance(
                batch_data, torch.Tensor
            ):
                if "input_ids" in batch_data:
                    input_ids_original = batch_data["input_ids"]
                if "attention_mask" in batch_data:
                    attention_mask_original = batch_data["attention_mask"]
            else:
                raise ValueError("input_ids and attention_mask must be provided")
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                model_tok,
                fix_tok,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(
                input_ids_original, attention_mask_original
            )

        return (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            model_tok,
            fix_tok,
            sentences,
        )

    def _compute_mapped_fixations_verbose(self, batch_data):
        input_ids_original = batch_data
        if self.version == 1:
            if isinstance(batch_data, Iterable) and not isinstance(
                batch_data, torch.Tensor
            ):
                if "input_ids" in batch_data:
                    input_ids_original = batch_data["input_ids"]
            (
                fixations,
                _,
                mapped_fixations_corrected,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
                tokens_idx_mapped,
                tokens_id_mapped,
                words_str_mapped,
                tokens_str_mapped,
            ) = self.FP_model._compute_mapped_fixations(
                input_ids_original, return_all=True
            )
            return (
                fixations,
                _,
                mapped_fixations_corrected,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
                tokens_idx_mapped,
                tokens_id_mapped,
                words_str_mapped,
                tokens_str_mapped,
            )

    def process_data(self, dataloader: torch.utils.data.DataLoader):
        fixations_all = []
        for batch in dataloader:
            # print("-----------------------------------------------")
            # print(batch)
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                model_tok,
                fix_tok,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(batch["input_ids"])
            fixations_all.append(
                {
                    "fixations": fixations,
                    "fixations_attention_mask": fixations_attention_mask,
                    "mapped_fixations": mapped_fixations,
                    "fix_tok": fix_tok,
                    "model_tok": model_tok,
                    "sentences": sentences,
                }
            )

        return fixations_all

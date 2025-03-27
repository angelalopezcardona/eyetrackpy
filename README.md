# EyeTrackPy

EyeTrackPy is a Python library for eye tracking analysis and prediction. The library consists of three main modules and includes example implementations.

## Modules

### 1. Data Generator
#### fixations_predictor:
This module includes a class for using two models to generate reading metrics (features) from text. Both models are based on two papers that we mention below. One of the main issues with these generative models is that they predict reading measures per token, and there tokens are different depending on the model they use. Our tool allows mapping these fixations to another selected model. Examples of how to perform this mapping can be found in the examples folder.
It can integrate fixations_predictor_trained_1 and fixations_predictor_trained_2.

First, we perform an initial mapping of tokens to the words they belong to in each tokenizer with some properties of \textit{FastTokenizers} from the \textit{transformers} library. Then, we map words from one tokenizer to the words in the other and finally, we assume that the combination of the tokens that are mapped to a word in one tokenizer correspond to the tokens that are mapped to the word that is mapped to the initial word in the other tokenizer. 

For each predictor, we reverse the method used to convert word-level features into token-level features but passing from tokens in the first one, to tokens in the second tokenizer. For example, if for the first \acrshort{et} features predictor models tokens $t_{1},t_{2}$ are mapped to tokens $t_{1},t_{2},t_{3}$ in another second tokenizer, the values sum for all the tokens in the first list and distribute them equally across all the tokens in the second list: being $t_{1}$ (1s TRT) and $t_{2}$ (2s TRT) each of $t_{1},t_{2},t_{3}$ are assigned a TRT of $(1+2)/3=1s$.  
To be able to map between tokenizers you need to install:
```sh
pip install git+https://github.com/anlopez94/tokenizer_aligner.git@v1.0.0
```
**Token Mapping Example**

Example of mapping Total Reading Time (TRT) between two different tokenizers:

| Field | Tokenizer 1 | Tokenizer 2 |
|-------|-------------|-------------|
| Words | astrophotography | astrophotography |
| Tokens | ['_Astro', 'photo', 'graphy'] | ['Ċ', 'Ast', 'roph', 'ot', 'ography'] |
| Token indices | [22, 23, 24] | [23, 24, 25, 26, 271] |
| Token IDs | [15001, 17720, 16369] | [198, 62152, 22761, 354, 5814] |
| TRT (1) | [11.23, 11.49, 10.16] | [6.58, 6.58, 6.58, 6.58, 6.58] |
| TRT (2) | [24.53, 0, 0] | [24.53, 0, 0, 0, 0] |

Note:
- TRT (1): Process used for the first eye-tracking predictor
- TRT (2): Process used for the second eye-tracking predictor

#### fixations_predictor_trained_1: Text Total Reading Time (TRT) Predictor
- Predicts the total reading time for text based on the paper cited below
- Uses eye-tracking data to identify important tokens for language modeling
- Pre-trained weights are automatically downloaded on first use
If you use this model, please cite:
```bibtex
@inproceedings{huang2023long,
  title={Long-Range Language Modeling with Selective Cache},
  author={Huang, Xuanli and Hollenstein, Nora},
  booktitle={Findings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={4838--4858},
  year={2023}
}
```
#### fixations_predictor_trained_2: Text Reading Measures Predictor
- Predicts 5 key reading measures
- Based on the TorontoCL model from CMCL 2021 Shared Task
- Uses RoBERTa with multi-stage fine-tuning
- Pre-trained on Provo corpus and fine-tuned on task-specific data
- Pre-trained weights are automatically downloaded on first use

If you use this model, please cite:
```bibtex
@inproceedings{li_torontocl_2021,
    title = {{TorontoCL} at {CMCL} 2021 {Shared} {Task}: {RoBERTa} with {Multi}-{Stage} {Fine}-{Tuning} for {Eye}-{Tracking} {Prediction}},
    author = {Li, Bai and Rudzicz, Frank},
    booktitle = {Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics},
    year = {2021},
    pages = {85--89},
    publisher = {Association for Computational Linguistics},
    doi = {10.18653/v1/2021.cmcl-1.9},
    url = {https://aclanthology.org/2021.cmcl-1.9}
}
```

#### fixations_predictor_trained_mdsem: Image Saliency Predictor (MD-SEM)
- Predicts visual importance and attention patterns across graphic design images
- Multi-duration saliency estimation model based on human eye-tracking data
- Generates predictions for different viewing durations (500ms, 3000ms, 5000ms)
- Pre-trained weights are automatically downloaded on first use

If you use this model, please cite:
```bibtex
@inproceedings{fosco_predicting_2020,
	address = {Virtual Event USA},
	title = {Predicting {Visual} {Importance} {Across} {Graphic} {Design} {Types}},
	language = {en},
	booktitle = {Proceedings of the 33rd {Annual} {ACM} {Symposium} on {User} {Interface} {Software} and {Technology}},
	publisher = {ACM},
	author = {Fosco, Camilo and Casser, Vincent and Bedi, Amish Kumar and O'Donovan, Peter and Hertzmann, Aaron and Bylinskii, Zoya},
	month = oct,
	year = {2020},
}
```
#### Pre-trained Weights
All model weights are automatically downloaded when you first use each model. The weights are:
- Stored in their respective model directories
- Added to `.gitignore` to prevent large files in the repository

### 2. Data Processor
Tools for processing and analyzing eye tracking data, including:
- Words straction from images
- Algorithms to asign fixations to words

### 3. Data printer
Visualization tools for:
- Scanpath visualization
- Reading measures over text
- Saliency map 

## Examples
The `examples` folder contains practical implementations demonstrating how to:
- Generate saliency maps on images
```bash
cd examples/data_generator
python main_predict_saliency_mdsem.py 
```
- Generate fixations on text
```bash
cd examples/data_generator
python main_predict_fixations_words.py 
```
- Process eye tracking data from experiments
- Visualize scanpath on images
```bash
cd examples/data_printer
python main_plot_scanpaths.py
```

- Visualize saliency maps on images
```bash
cd examples/data_printer
python main_plot_saliency.py
```
- Visualize reading metrics on text
```bash
cd examples/data_printer
python main_plot_fixations_words.py
```

## Installation

```sh
pip install git+https://github.com/anlopez94/eyetrackpy.git
```


## Requirements
Requirements vary by module:

Core requirements:
- Python 3.7+
- NumPy
- Matplotlib
- PIL

Data Generator module:
- TensorFlow (for MDSEM model)
- PyTorch (for fixation prediction models)
- OpenCV
- To be able to map fixations, run the following command:

```sh
pip install git+https://github.com/anlopez94/tokenizer_aligner.git
```

Data Processor & Printer modules:
- OpenCV




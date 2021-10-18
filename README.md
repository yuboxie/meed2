# Empathetic Dialog Generation with Fine-Grained Intents

Code for paper
> Yubo Xie and Pearl Pu. Empathetic Dialog Generation with Fine-Grained Intents.
> CoNLL 2021. [PDF Link](https://arxiv.org/pdf/2105.06829.pdf).

## Environment
The project was developed using the following packages:

    tqdm==4.49.0
    numpy==1.19.3
    scipy==1.5.2
    pandas==1.1.0
    tensorflow==2.3.1
    pytorch_transformers==1.2.0

## Files
- `datasets.py`: read the data and tokenize the text;
- `model_basics.py`: implementation of Transformer basic components;
- `model_emo_pred.py`: implementation of the response emotion/intent predictor;
- `model.py`: implementation of the empathetic dialog model;
- `model_utils.py`: utility functions for the model implementation;
- `train_os.py`: pre-train the model on the OS dataset;
- `train_edos.py`: fine-tune the model on the EDOS dataset;
- `train_ed.py`: fine-tune the model on the ED dataset;
- `train_emo_os.py`: pre-train the response emotion/intent predictor on the OS dataset;
- `train_emo_edos.py`: fine-tune the response emotion/intent predictor on the EDOS dataset;
- `train_emo_ed.py`: fine-tune the response emotion/intent predictor on the ED dataset;
- `predict_emo.py`: predict the response emotion/intent;
- `beam_search.py`: implementation of the beam search algorithm;
- `predict.py`: generate the responses.

## Trained Models
TensorFlow checkpoints can be found [here](https://drive.google.com/drive/folders/1n1MSVwn9ud1lfGgif2yIedPecFrmsmCV?usp=sharing).

## Datasets
The OS and EDOS datasets can be found [here](https://drive.google.com/drive/folders/16-dkORqc6p7q5j14zNN_t7_V-NJjn6ga?usp=sharing).

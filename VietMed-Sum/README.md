# Real-time Speech Summarization for Medical Conversations

**<div align="center">Interspeech 2024 (Oral)</div>**

> Please press ⭐ button and/or cite papers if you feel helpful.

* **Abstract:**
In doctor-patient conversations, identifying medically relevant information is crucial, posing the need for conversation summarization. In this work, we propose the first deployable real-time speech summarization system for real-world applications in industry, which generates a local summary after every N speech utterances within a conversation and a global summary after the end of a conversation. Our system could enhance user experience from a business standpoint, while also reducing computational costs from a technical perspective. Secondly, we present VietMed-Sum which, to our knowledge, is the first speech summarization dataset for medical conversations. Thirdly, we are the first to utilize LLM and human annotators collaboratively to create gold standard and synthetic summaries for medical conversation summarization. Finally, we present baseline results of state-of-the-art models on VietMed-Sum. All code, data (English-translated and Vietnamese) and models are available online.

* **Citation:**
Please cite this paper: https://arxiv.org/abs/2406.15888

``` bibtex
@article{VietMed_Sum,
    title={Real-time Speech Summarization for Medical Conversations},
    author={Le-Duc, Khai and Nguyen, Khai-Nguyen and Vo-Dang, Long and Hy, Truong-Son},
    journal={arXiv preprint arXiv:2406.15888},
    booktitle={Interspeech 2024},
    url = {https://arxiv.org/abs/2406.15888},
    year={2024}
    }
```

This repository contains scripts for automatic speech recognition (ASR) and real-time speech summarization (RTSS) using sequence-to-sequence (seq2seq) models. The provided scripts cover model preparation, training, inference, and evaluation processes, based on the dataset VietMed-Sum.

<p align="center">
<img src="/VietMed-Sum/RTSS_diagram.png" alt="drawing" width="900"/>
</p>

## Dataset and Pre-trained Models:

Dataset: [HuggingFace dataset](https://huggingface.co/datasets/leduckhai/VietMed-Sum), [Paperswithcodes dataset](https://paperswithcode.com/dataset/vietmed-sum)

Pre-trained models: [HuggingFace model](https://huggingface.co/leduckhai/ViT5-VietMedSum)

## For reproducing experiments:
Data for train, dev, test is in the corresponding folder. Each folder contains different split of the dataset. The ASR data for testing is in ./test/test_asr.xlsx

To train the model, run ./run.sh for automated running (assuming you have 8 GPUs).

To do inference, run ./run_inference.sh

## For infering using HuggingFace:

Install the pre-requisite packages in Python. 
```python
pip install transformers
```

Use the code below to get started with the model.

```python
from transformers import pipeline
# Initialize the pipeline with the ViT5 model, specify the device to use CUDA for GPU acceleration
pipe = pipeline("text2text-generation", model="monishsystem/medisum_vit5", device='cuda')
# Example text in Vietnamese describing a traditional medicine product
example = "Loại thuốc này chứa các thành phần đông y đặc biệt tốt cho sức khoẻ, giúp tăng cường sinh lý và bổ thận tráng dương, đặc biệt tốt cho người cao tuổi và người có bệnh lý nền"
# Generate a summary for the input text with a maximum length of 50 tokens
summary = pipe(example, max_new_tokens=50)
# Print the generated summary
print(summary)
```

## Contact:

Core developers:

**Khai Le-Duc**
```
University of Toronto, Canada
Email: duckhai.le@mail.utoronto.ca
GitHub: https://github.com/leduckhai
```

**Khai-Nguyen Nguyen**
```
College of William & Mary, USA
Email: knguyen07@wm.edu
GitHub: https://github.com/nkn002
```

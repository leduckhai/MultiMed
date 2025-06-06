# MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder

**<div align="center">ACL 2025</div>**

<div align="center"><b>Khai Le-Duc</b>, Phuc Phan, Tan-Hanh Pham, Bach Phan Tat,</div>

<div align="center">Minh-Huong Ngo, Chris Ngo, Thanh Nguyen-Tang, Truong-Son Hy</div>


> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
  <img src="https://github.com/leduckhai/MultiMed/blob/master/MultiMed/MultiMed_ACL2025.png" width="700"/>
</p>

* **Abstract:**
Multilingual automatic speech recognition (ASR) in the medical domain serves as a foundational task for various downstream applications such as speech translation, spoken language understanding, and voice-activated assistants. This technology improves patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we introduce \textit{MultiMed}, the first multilingual medical ASR dataset, along with the first collection of small-to-large end-to-end medical ASR models, spanning five languages: Vietnamese, English, German, French, and Mandarin Chinese. To our best knowledge, \textit{MultiMed} stands as **the world’s largest medical ASR dataset across all major benchmarks**: total duration, number of recording conditions, number of accents, and number of speaking roles. Furthermore, we present the first multilinguality study for medical ASR, which includes reproducible empirical baselines, a monolinguality-multilinguality analysis, Attention Encoder Decoder (AED) vs Hybrid comparative study and a linguistic analysis. We present practical ASR end-to-end training schemes optimized for a fixed number of trainable parameters that are common in industry settings. All code, data, and models are available online: [https://github.com/leduckhai/MultiMed/tree/master/MultiMed](https://github.com/leduckhai/MultiMed/tree/master/MultiMed).

* **Citation:**
Please cite this paper: [https://arxiv.org/abs/2409.14074](https://arxiv.org/abs/2409.14074)

``` bibtex
@article{le2024multimed,
  title={MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder},
  author={Le-Duc, Khai and Phan, Phuc and Pham, Tan-Hanh and Tat, Bach Phan and Ngo, Minh-Huong and Ngo, Chris and Nguyen-Tang, Thanh and Hy, Truong-Son},
  journal={arXiv preprint arXiv:2409.14074},
  year={2024}
}
```

This repository contains scripts for medical automatic speech recognition (ASR) for 5 languages: Vietnamese, English, German, French, and Mandarin Chinese. 
The provided scripts cover model preparation, training, inference, and evaluation processes, based on the dataset *MultiMed*.

## Dataset and Pre-trained Models:

Dataset: [🤗 HuggingFace dataset](https://huggingface.co/datasets/leduckhai/MultiMed), [Paperswithcodes dataset](https://paperswithcode.com/dataset/multimed)

Pre-trained models: [🤗 HuggingFace models](https://huggingface.co/leduckhai/MultiMed)

| Model Name       | Description                                | Link                                                                 |
|------------------|--------------------------------------------|----------------------------------------------------------------------|
| `Whisper-Small-Chinese`     | Small model fine-tuned on medical Chinese set        | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-chinese) |
| `Whisper-Small-English`    | Small model fine-tuned on medical English set         | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-english) |
| `Whisper-Small-French`  | Small model fine-tuned on medical French set          | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-french)    |
| `Whisper-Small-German`  | Small model fine-tuned on medical German set          | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-german)    |
| `Whisper-Small-Vietnamese`  | Small model fine-tuned on medical Vietnamese set          | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-vietnamese)    |
| `Whisper-Small-Multilingual`  | Small model fine-tuned on medical Multilingual set (5 languages)        | [Hugging Face models](https://huggingface.co/leduckhai/MultiMed-ST/tree/main/asr/whisper-small-multilingual)    |


## Contact:

Core developers:

**Khai Le-Duc**
```
University of Toronto, Canada
Email: duckhai.le@mail.utoronto.ca
GitHub: https://github.com/leduckhai
```

**Phuc Phan**
```
FPT University, Vietnam
LinkedIn: https://www.linkedin.com/in/pphuc/
```

**Tan-Hanh Pham**
```
Florida Institute of Technology, USA
Email: tpham2023@my.fit.edu
GitHub: https://github.com/Hanhpt23/
```

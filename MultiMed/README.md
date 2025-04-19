# MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder

**<div align="center">Preprint</div>**

<div align="center">Khai Le-Duc, Phuc Phan, Tan-Hanh Pham, Bach Phan Tat, Minh-Huong Ngo, Thanh Nguyen-Tang, Truong-Son Hy</div>


> Please press ⭐ button and/or cite papers if you feel helpful.

* **Abstract:**
Multilingual automatic speech recognition (ASR) in the medical domain serves as a foundational task for various downstream applications such as speech translation, spoken language understanding, and voice-activated assistants. This technology enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we introduce MultiMed, the first multilingual medical ASR dataset, along with the first collection of small-to-large end-to-end medical ASR models, spanning five languages: Vietnamese, English, German, French, and Mandarin Chinese. To our best knowledge, MultiMed stands as the world’s largest medical ASR dataset across all major benchmarks: total duration, number of recording conditions, number of accents, and number of speaking roles. Furthermore, we present the first multilinguality study for medical ASR, which includes reproducible empirical baselines, a monolinguality-multilinguality analysis, Attention Encoder Decoder (AED) vs Hybrid comparative study, a layer-wise ablation study for the AED, and a linguistic analysis for multilingual medical ASR. All code, data, and models are available online: [https://github.com/leduckhai/MultiMed/tree/master/MultiMed](https://github.com/leduckhai/MultiMed/tree/master/MultiMed).

* **Citation:**
Please cite this paper: [https://arxiv.org/abs/2409.14074](https://arxiv.org/abs/2409.14074)

``` bibtex
@article{le2024multimed,
  title={MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder},
  author={Le-Duc, Khai and Phan, Phuc and Pham, Tan-Hanh and Tat, Bach Phan and Ngo, Minh-Huong and Hy, Truong-Son},
  journal={arXiv preprint arXiv:2409.14074},
  year={2024}
}
```

This repository contains scripts for medical automatic speech recognition (ASR) for 5 languages: Vietnamese, English, German, French, and Mandarin Chinese. 
The provided scripts cover model preparation, training, inference, and evaluation processes, based on the dataset *MultiMed*.

## Dataset and Pre-trained Models:

Dataset: [🤗 HuggingFace dataset](https://huggingface.co/datasets/leduckhai/MultiMed), [Paperswithcodes dataset](https://paperswithcode.com/dataset/multimed)

Pre-trained models: [🤗 HuggingFace models](https://huggingface.co/leduckhai/MultiMed)

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

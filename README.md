# MultiMed: Multilingual Multitask Multipurpose Medical Speech Recognition

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-03.07.2024-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Le%20Duc%20Khai-pink?style=for-the-badge"> 
</p>

<p align="center">
<img src="https://img.shields.io/badge/Speech Recognition-white"> 
<img src="https://img.shields.io/badge/Natural Language Processing-white">
<img src="https://img.shields.io/badge/Large Language Model-white">     
<img src="https://img.shields.io/badge/Deep Learning-white">      
<img src="https://img.shields.io/badge/AI for Healthcare-white">
</p>

## Code for implemented papers

<details><summary>VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain (LREC-COLING 2024, Oral) </summary><p>

* [Code](VietMed/README.md)

* **Abstract:**
Due to privacy restrictions, there's a shortage of publicly available speech recognition datasets in the medical domain. In this work, we present VietMed - a Vietnamese speech recognition dataset in the medical domain comprising 16h of labeled medical speech, 1000h of unlabeled medical speech and 1200h of unlabeled general-domain speech. To our best knowledge, VietMed is by far the world's largest public medical speech recognition dataset in 7 aspects: total duration, number of speakers, diseases, recording conditions, speaker roles, unique medical terms and accents. VietMed is also by far the largest public Vietnamese speech dataset in terms of total duration. Additionally, we are the first to present a medical ASR dataset covering all ICD-10 disease groups and all accents within a country. Moreover, we release the first public large-scale pre-trained models for Vietnamese ASR, w2v2-Viet and XLSR-53-Viet, along with the first public large-scale fine-tuned models for medical ASR. Even without any medical data in unsupervised pre-training, our best pre-trained model XLSR-53-Viet generalizes very well to the medical domain by outperforming state-of-the-art XLSR-53, from 51.8% to 29.6% WER on test set (a relative reduction of more than 40%). All code, data and models are made publicly available here: [https://github.com/leduckhai/MultiMed/tree/master/VietMed](https://github.com/leduckhai/MultiMed/tree/master/VietMed)
    
* **Citation:**
Please cite this paper https://arxiv.org/abs/2404.05659

``` bibtex
@inproceedings{le2024vietmed,
  title={VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain},
  author={Le-Duc, Khai},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={17365--17370},
  year={2024}
}
```
</p></details>

<details><summary>Medical Spoken Named Entity Recognition (NAACL 2025) </summary><p>
    
* [Code](VietMed-NER/README.md)

* **Abstract:**
Spoken Named Entity Recognition (NER) aims to extract named entities from speech and categorise them into types like person, location, organization, etc. In this work, we present VietMed-NER - the first spoken NER dataset in the medical domain. To our knowledge, our Vietnamese real-world dataset is the largest spoken NER dataset in the world regarding the number of entity types, featuring 18 distinct types. Furthermore, we present baseline results using various state-of-the-art pre-trained models: encoder-only and sequence-to-sequence; and conduct quantitative and qualitative error analysis. We found that pre-trained multilingual models generally outperform monolingual models on reference text and ASR output and encoders outperform sequence-to-sequence models in NER tasks. By translating the transcripts, the dataset can also be utilised for text NER in the medical domain in other languages than Vietnamese. All code, data and models are publicly available: [https://github.com/leduckhai/MultiMed/tree/master/VietMed-NER](https://github.com/leduckhai/MultiMed/tree/master/VietMed-NER)

* **Citation:**
Please cite this paper https://arxiv.org/abs/2406.13337

``` bibtex
@article{le2024medical,
  title={Medical Spoken Named Entity Recognition},
  author={Le-Duc, Khai and Thulke, David and Tran, Hung-Phong and Vo-Dang, Long and Nguyen, Khai-Nguyen and Hy, Truong-Son and Schl{\"u}ter, Ralf},
  journal={arXiv preprint arXiv:2406.13337},
  year={2024}
}
```
</p></details>

<details><summary> Real-time Speech Summarization for Medical Conversations (Interspeech 2024, Oral) </summary><p>

<p align="center">
<img src="/VietMed-Sum/RTSS_diagram.png" alt="drawing" width="900"/>
</p>

* [Code](VietMed-Sum/README.md)

* **Abstract:**
In doctor-patient conversations, identifying medically relevant information is crucial, posing the need for conversation summarization. In this work, we propose the first deployable real-time speech summarization system for real-world applications in industry, which generates a local summary after every N speech utterances within a conversation and a global summary after the end of a conversation. Our system could enhance user experience from a business standpoint, while also reducing computational costs from a technical perspective. Secondly, we present VietMed-Sum which, to our knowledge, is the first speech summarization dataset for medical conversations. Thirdly, we are the first to utilize LLM and human annotators collaboratively to create gold standard and synthetic summaries for medical conversation summarization. Finally, we present baseline results of state-of-the-art models on VietMed-Sum. All code, data (English-translated and Vietnamese) and models are available online: [https://github.com/leduckhai/MultiMed/tree/master/VietMed-Sum](https://github.com/leduckhai/MultiMed/tree/master/VietMed-Sum)

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
</p></details>

**Below is work in progress, will be available soon!**

<details><summary>Two-Stage Intermediate Loss for Fine-tuning Self-Supervised Models</summary><p>
Due to the double-blind review, request of implementation and models will be processed after paper notification.
</p></details>

<details><summary>Domain-Shift in Medical Machine Translation</summary><p>
Due to the double-blind review, request of implementation and models will be processed after paper notification.
</p></details>

## Contact:

For any information, please contact the main author:

Le Duc Khai at University of Toronto, Canada

Email: duckhai.le@mail.utoronto.ca

GitHub: https://github.com/leduckhai

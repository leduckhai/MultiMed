# VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain 

**<div align="center">LREC-COLING 2024 (Oral)</div>**

<div align="center">Khai Le-Duc</div>

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

## Dataset and Pre-trained Models:

To load labeled data, please refer to our:

🤗 [HuggingFace](https://huggingface.co/datasets/leduckhai/VietMed)

[Paperswithcodes](https://paperswithcode.com/dataset/vietmed).

For full dataset (labeled data + unlabeled data) and pre-trained models, please refer to [Google Drive](https://drive.google.com/drive/folders/1hsoB_xjWh66glKg3tQaSLm4S1SVPyANP?usp=sharing)


## Reproduce Experiments:
Please check "config" folder for reproducibility.

Necessary packages for GMM-HMM ASR: [RETURNN](https://github.com/rwth-i6/returnn), [Sisyphus](https://github.com/rwth-i6/sisyphus), [RASR](https://github.com/rwth-i6/rasr), [SRILM](http://www.speech.sri.com/projects/srilm/), [Fairseq](https://github.com/facebookresearch/fairseq).

You may also want to check how to fine-tune our wav2vec 2.0-based pre-trained models [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md).

## Contact:

If any links are broken, please contact me for fixing!

```
Le Duc Khai
University of Toronto, Canada
Email: duckhai.le@mail.utoronto.ca
GitHub: https://github.com/leduckhai
```

# Original import
from sisyphus import gs

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lexicon.conversion import LexiconToWordListJob
from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
#from i6_experiments.common.datasets.tedlium2.lexicon import get_g2p_augmented_bliss_lexicon
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH
from i6_experiments.common.setups.lm.srilm_system import SriLmSystem
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter
from i6_core.text import ConcatenateJob

# New import
import os.path
from sisyphus import gs, tk, Path
    
    
def run_VietMed_ngram_lm(add_unknown_phoneme_and_mapping: bool = False, alias_prefix="vietmed_v1_lm"):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    train_data = CorpusToTxtJob(Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_train_v1.xml.gz", cached=True)).out_txt
    dev_data = CorpusToTxtJob(Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_dev_v1.xml.gz", cached=True)).out_txt
    test_data = CorpusToTxtJob(Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_test_v1.xml.gz", cached=True)).out_txt
    extra_train_data = ConcatenateJob(text_files=[
        # Text data
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/BABEL_vi_merged_corpus.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/CommonVoice_vi_merged_corpus.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/FPTOpenSpeech_merged_corpus.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/PhoNER_Covid19_merged.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/ViHealthBERT_FAQ_merged.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/VIVOS_merged_corpus.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/VLSP2020_merged_corpus.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Text_corpus/VNTC_Health.txt", cached=True),
        # Extra vocab
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Vocab/ICD10_medicalvocab_27k.txt", cached=True),
        Path("/work/asr3/hykist/data/vietnamese/monolingual_text/Vocab/Vietnamese_vocab_74k.txt", cached=True),
        ]).out
    #train_data_dict = get_text_data_dict()
    train_data_dict = {"audio-transcriptions": train_data, "background-data": extra_train_data} # train_data_dict can have multiple keys, each key represents a text dataset?
    
    print()
    print('------------------------------DEBUG------------------------------')
    print(train_data_dict)
    print()
    
    dev_data_dict = {"dev": dev_data}
    test_data_dict = {
        "dev": dev_data,
        "test": test_data,
    }

    '''
    vocab = LexiconToWordListJob(
        get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping, output_prefix="lexicon"
        )
    ).out_word_list
    '''
    # Seed lexicon or (should be) augmented lexicon
    vocab = LexiconToWordListJob(
        Path("/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz", cached=True)
    ).out_word_list

    ngram_system = SriLmSystem(
        name="vietmed_v1",
        train_data=train_data_dict,
        dev_data=dev_data_dict,
        eval_data=test_data_dict,
        ngram_order=[3, 4, 5],
        vocab=vocab,
        ngram_args=[
            "-gt1min 1",
            "-gt2min 1",
            "-gt3min 1",
            "-gt4min 1",
            "-gt5min 1",
            "-gt6min 1",
            "-interpolate",
            "-kndiscount",
        ],
        perplexity_args="-debug 2",
        srilm_path=SRILM_PATH,
        ngram_rqmt=None,
        perplexity_rqmt=None,
    )
    ngram_system.run_training()

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir
    return ngram_system
    
    
run_VietMed_ngram_lm()
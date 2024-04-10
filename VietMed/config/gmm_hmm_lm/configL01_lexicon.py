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
from i6_core.corpus.stats import ExtractOovWordsFromCorpusJob

import os.path
from sisyphus import gs, tk, Path


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


def get_g2p_augmented_bliss_lexicon(
    add_unknown_phoneme_and_mapping: bool = False,
    audio_format: str = "wav",
    output_prefix: str = "datasets",
) -> tk.Path:
    """
    augment the kernel lexicon with unknown words from the training corpus

    :param add_unknown_phoneme_and_mapping: add an unknown phoneme and mapping unknown phoneme:lemma
    :param audio_format: options: wav, ogg, flac, sph, nist. nist (NIST sphere format) and sph are the same.
    :param output_prefix:
    :return:
    """
    # Lexicon here is augmented with VietMed train corpus
    # Should be augmented with larger text datasets by creating a dummy corpus which includes all text data
    original_bliss_lexicon = Path("/u/dle/VietMed_05-09-2023/data_dump/babel_scripted_conv.lexicon", cached=True)
    corpus_name = "train"
    bliss_corpus = Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_train_v1.xml.gz", cached=True) 

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=original_bliss_lexicon,
        train_lexicon=original_bliss_lexicon,
    )
    augmented_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=bliss_corpus,
        corpus_name=corpus_name,
        alias_path=os.path.join(output_prefix, "g2p"),
        casing="lower",
    )

    return augmented_bliss_lexicon


vocab = LexiconToWordListJob(
        get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=False, output_prefix="lexicon"
        ))
vocab.add_alias("g2p_augmented_vocab")
tk.register_output("g2p_augmented_vocab", vocab.out_word_list)


# ----------------------My modification----------------------
# Get OOV of augmented lexicon for dev and test set
oov_dev = ExtractOovWordsFromCorpusJob(
    bliss_corpus=Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_dev_v1.xml.gz"),
    bliss_lexicon=get_g2p_augmented_bliss_lexicon(add_unknown_phoneme_and_mapping=False, output_prefix="lexicon"),
    casing="lower")
oov_dev.add_alias("extract_oov_dev")
tk.register_output("extract_oov_dev", oov_dev.out_oov_words)


oov_test = ExtractOovWordsFromCorpusJob(
    bliss_corpus=Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_test_v1.xml.gz"),
    bliss_lexicon=get_g2p_augmented_bliss_lexicon(add_unknown_phoneme_and_mapping=False, output_prefix="lexicon"),
    casing="lower")
oov_test.add_alias("extract_oov_test")
tk.register_output("extract_oov_test", oov_test.out_oov_words)
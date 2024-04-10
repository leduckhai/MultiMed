from sisyphus import gs

from i6_experiments.common.setups.rasr import gmm_system
from i6_experiments.common.setups.rasr.util import RasrSteps, OutputArgs

from i6_experiments.common.baselines.tedlium2.gmm import baseline_args
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs

#from ..default_tools import RASR_BINARY_PATH # Here is where to define which RASR version to use
from i6_experiments.common.baselines.tedlium2.default_tools import RASR_BINARY_PATH


from collections import defaultdict
from typing import Dict

from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2.corpus import get_corpus_object_dict
#from i6_experiments.common.datasets.tedlium2.lexicon import (
#    get_g2p_augmented_bliss_lexicon,
#)
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter
from i6_experiments.common.setups.rasr.util import RasrDataInput

from i6_experiments.common.setups.rasr.config.lex_config import (
    LexiconRasrConfig,
)
from i6_experiments.common.setups.rasr.config.lm_config import ArpaLmRasrConfig
from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

from i6_core.meta import CorpusObject
#from .constants import DURATIONS
from i6_experiments.common.datasets.tedlium2.constants import DURATIONS
from i6_core.features.filterbank import filter_width_from_channels
from i6_experiments.common.setups.rasr import util
from i6_experiments.common.baselines.librispeech.default_tools import SCTK_BINARY_PATH

import os.path
from sisyphus import gs, tk, Path
    
    
def get_corpus_object_dict() -> Dict[str, CorpusObject]:
    corpus_object_dict = {}
    
    # Train data
    corpus_object = CorpusObject()
    corpus_object.corpus_file = Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_train_v1.xml.gz", cached=True) 
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = DURATIONS["train"]

    corpus_object_dict["train"] = corpus_object
    
    # Dev data
    corpus_object = CorpusObject()
    corpus_object.corpus_file = Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_dev_v1.xml.gz", cached=True) 
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = DURATIONS["dev"]

    corpus_object_dict["dev"] = corpus_object
    
    # Test data
    corpus_object = CorpusObject()
    corpus_object.corpus_file = Path("/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_test_v1.xml.gz", cached=True) 
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = DURATIONS["test"]

    corpus_object_dict["test"] = corpus_object

    return corpus_object_dict
    
    
def get_corpus_data_inputs(add_unknown_phoneme_and_mapping: bool = True) -> Dict[str, Dict[str, RasrDataInput]]:
    corpus_object_dict = get_corpus_object_dict()

    train_lexicon = LexiconRasrConfig(
        #get_g2p_augmented_bliss_lexicon(
        #    add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping, output_prefix="lexicon"
        #),
        Path("/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz", cached=True), # Should be augmented lexicon
        False,
    )
    '''
    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    comb_lm = ArpaLmRasrConfig(lm_path=lm.ngram_lm)
    '''
    comb_lm = ArpaLmRasrConfig(lm_path=Path("/u/dle/VietMed_05-09-2023/data_dump/pruned_4gram_lm.gz", cached=True)) 
    
    rasr_data_input_dict = defaultdict(dict)

    for name, crp_obj in corpus_object_dict.items():
        rasr_data_input_dict[name][name] = RasrDataInput(
            corpus_object=crp_obj,
            lexicon=train_lexicon.get_dict(),
            #concurrent=CONCURRENT[name],
            concurrent=10,
            lm=comb_lm.get_dict() if name == "dev" or name == "test" else None,
        )

    return rasr_data_input_dict
    

def get_rasr_init_args():
    samples_options = {
        "audio_format": "wav",
        "dc_detection": False,
    }

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global",
        "nonword_phones": "",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            6.0,
        ),  # only used when tying_type = global-and-nonword
        #"allow_zero_weights": True, # To avoid zero weight in RASR
    }

    costa_args = {"eval_recordings": True, "eval_lm": False}

    feature_extraction_args = {
        "mfcc": {
            "num_deriv": 2,
            "num_features": None,  # confusing name: number of max features, above number -> clipped
            "mfcc_options": {
                "warping_function": "mel",
                # to be compatible with our old magic number, we have to use 20 features
                "filter_width": filter_width_from_channels(channels=20, warping_function="mel", f_max=8000),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": samples_options,
                "cepstrum_options": {
                    "normalize": False,
                    "outputs": 16,  # this is the actual output feature dimension
                    "add_epsilon": True,  # when there is no dc-detection we can have log(0) otherwise
                    "epsilon": 1e-10,
                },
                "fft_options": None,
                "add_features_output": True,
            },
        },
        "energy": {
            "energy_options": {
                "without_samples": False,
                "samples_options": samples_options,
                "fft_options": None,
            }
        },
    }

    scorer_args = {"sctk_binary_path": SCTK_BINARY_PATH}

    return util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer_args=scorer_args,
    )
        
def run_VietMed_common_baseline(
    alias_prefix="vietmed_gmm",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    #rasr_init_args = baseline_args.get_init_args()
    rasr_init_args = get_rasr_init_args()
    mono_args = baseline_args.get_monophone_args()
    cart_args = baseline_args.get_cart_args()
    tri_args = baseline_args.get_triphone_args()
    vtln_args = baseline_args.get_vtln_args()
    sat_args = baseline_args.get_sat_args()
    vtln_sat_args = baseline_args.get_vtln_sat_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("train", "train")
    final_output_args.define_corpus_type("dev", "dev")
    final_output_args.define_corpus_type("test", "test")
    # final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)
    steps.add_step("cart", cart_args)
    steps.add_step("tri", tri_args)
    steps.add_step("vtln", vtln_args)
    steps.add_step("sat", sat_args)
    steps.add_step("vtln+sat", vtln_sat_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs()

    system = gmm_system.GmmSystem(rasr_binary_path=RASR_BINARY_PATH)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data["train"],
        dev_data=corpus_data["dev"],
        #test_data={},  # corpus_data["test"],
        test_data=corpus_data["test"],
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return system
    
    
run_VietMed_common_baseline()
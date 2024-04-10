"""
  Get recog system for VietMed 8kHz sets 
  Version data3
"""

import copy
import os.path
import pickle

from sisyphus import gs, tk, Path
from sisyphus.delayed_ops import DelayedFunction
import i6_core.am as am
import i6_core.corpus as corpus_recipes
import i6_core.features as features
import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
from i6_core.meta.system import CorpusObject
from i6_private.users.vieting.helpers.alignment_cropping import cropping, cropping_out_type
#from i6_private.users.vieting.helpers.wav2vec2 import network as wav2vec2_network
from i6_private.users.dle.XLSR_Peter_8enc import network as wav2vec2_base_network
from i6_private.users.vieting.helpers.xlsr import network as xlsr_network
from i6_private.users.vieting.pipeline.nn_training import train
from i6_private.users.vieting.pipeline.system import HybridRecognitionSystem
from returnn_common.asr import gt
    

# VietMed dev v1
def get_recog_system_VietMed_dev_v1(   
    wave_norm,    
    corpus="vietmed_corpus_dev_v1",
    corpus_file="/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_dev_v1.xml.gz",        
    lexicon="/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz",
    cart_tree="/u/dle/VietMed_05-09-2023/data_dump/cart.tree.xml.gz",
    allophone_file="/u/dle/VietMed_05-09-2023/data_dump/allophones",
    mixtures="/work/asr3/luescher/hiwis/dle/17-10-2022_VietMed/data3/am.mix", # Dummy, AdvancedTreeSearchLmImageAndGlobalCacheJob somehow reads another am.mix path
    lm_file="/u/dle/VietMed_05-09-2023/data_dump/pruned_4gram_lm.gz",
    lm_scale=10.0,
    concurrent=20,
):
    corpus_object = CorpusObject()
    corpus_object.audio_dir = "/work/asr3/hykist/data/vietnamese/8khz/VietMed/Audio_8kHz_wav/"
    corpus_object.audio_format = "wav"
    corpus_object.corpus_file = Path(corpus_file, cached=True)
    corpus_object.duration = 10

    recog_system = HybridRecognitionSystem()
    recog_system.crp["base"].acoustic_model_config = am.acoustic_model_config(
        **{
            "tdp_transition": (3.0, 0.0, 30.0, 0.0),
            "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        }
    )

    recog_system.set_corpus(corpus, corpus_object, concurrent)
    recog_system.create_stm_from_corpus(corpus)
    recog_system.set_sclite_scorer(corpus)
    
    if wave_norm == False:
        recog_system.feature_flows[corpus]["waveform_scaled"] = features.samples_flow(
            dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    else:
        recog_system.feature_flows[corpus]["waveform"] = features.samples_flow(
            dc_detection=False, input_options={"block-size": "1"})
            
    recog_system.mfcc_features(corpus, mfcc_options={"filter_width": 334.1203584362728})
    recog_system.crp[corpus].lexicon_config = rasr.RasrConfig()
    recog_system.crp[corpus].lexicon_config.file = Path(lexicon, cached=True)
    recog_system.crp[corpus].lexicon_config.normalize_pronunciation = False
    recog_system.crp[corpus].acoustic_model_config.state_tying.type = "cart"
    recog_system.crp[corpus].acoustic_model_config.state_tying.file = Path(cart_tree, cached=True,)
    recog_system.crp[corpus].acoustic_model_config.allophones.add_all = False
    recog_system.crp[corpus].acoustic_model_config.allophones.add_from_file = Path(allophone_file, cached=True)
    recog_system.crp[corpus].acoustic_model_config.allophones.add_from_lexicon = True
    recog_system.crp[corpus].language_model_config = rasr.RasrConfig()
    recog_system.crp[corpus].language_model_config.type = "ARPA"
    recog_system.crp[corpus].language_model_config.file = Path(lm_file, cached=True)
    recog_system.crp[corpus].language_model_config.scale = lm_scale

    recog_system.set_corpus("train", CorpusObject(), 1)  # dummy
    recog_system.mixtures["train"] = {"default": Path(mixtures, cached=True),}
    return recog_system   

       
# VietMed test v1
def get_recog_system_VietMed_test_v1(   
    wave_norm,    
    corpus="vietmed_corpus_test_v1",
    corpus_file="/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_test_v1.xml.gz",        
    lexicon="/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz",
    cart_tree="/u/dle/VietMed_05-09-2023/data_dump/cart.tree.xml.gz",
    allophone_file="/u/dle/VietMed_05-09-2023/data_dump/allophones",
    mixtures="/work/asr3/luescher/hiwis/dle/17-10-2022_VietMed/data3/am.mix", # Dummy, AdvancedTreeSearchLmImageAndGlobalCacheJob somehow reads another am.mix path
    lm_file="/u/dle/VietMed_05-09-2023/data_dump/pruned_4gram_lm.gz",
    lm_scale=10.0,
    concurrent=20,
):
    corpus_object = CorpusObject()
    corpus_object.audio_dir = "/work/asr3/hykist/data/vietnamese/8khz/VietMed/Audio_8kHz_wav/"
    corpus_object.audio_format = "wav"
    corpus_object.corpus_file = Path(corpus_file, cached=True)
    corpus_object.duration = 10

    recog_system = HybridRecognitionSystem()
    recog_system.crp["base"].acoustic_model_config = am.acoustic_model_config(
        **{
            "tdp_transition": (3.0, 0.0, 30.0, 0.0),
            "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        }
    )

    recog_system.set_corpus(corpus, corpus_object, concurrent)
    recog_system.create_stm_from_corpus(corpus)
    recog_system.set_sclite_scorer(corpus)
    
    if wave_norm == False:
        recog_system.feature_flows[corpus]["waveform_scaled"] = features.samples_flow(
            dc_detection=False, input_options={"block-size": "1"}, scale_input=2**-15)
    else:
        recog_system.feature_flows[corpus]["waveform"] = features.samples_flow(
            dc_detection=False, input_options={"block-size": "1"})
            
    recog_system.mfcc_features(corpus, mfcc_options={"filter_width": 334.1203584362728})
    recog_system.crp[corpus].lexicon_config = rasr.RasrConfig()
    recog_system.crp[corpus].lexicon_config.file = Path(lexicon, cached=True)
    recog_system.crp[corpus].lexicon_config.normalize_pronunciation = False
    recog_system.crp[corpus].acoustic_model_config.state_tying.type = "cart"
    recog_system.crp[corpus].acoustic_model_config.state_tying.file = Path(cart_tree, cached=True,)
    recog_system.crp[corpus].acoustic_model_config.allophones.add_all = False
    recog_system.crp[corpus].acoustic_model_config.allophones.add_from_file = Path(allophone_file, cached=True)
    recog_system.crp[corpus].acoustic_model_config.allophones.add_from_lexicon = True
    recog_system.crp[corpus].language_model_config = rasr.RasrConfig()
    recog_system.crp[corpus].language_model_config.type = "ARPA"
    recog_system.crp[corpus].language_model_config.file = Path(lm_file, cached=True)
    recog_system.crp[corpus].language_model_config.scale = lm_scale

    recog_system.set_corpus("train", CorpusObject(), 1)  # dummy
    recog_system.mixtures["train"] = {"default": Path(mixtures, cached=True),}
    return recog_system   
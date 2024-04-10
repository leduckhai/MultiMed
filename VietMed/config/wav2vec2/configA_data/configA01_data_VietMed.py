"""
  Config for VietMed data creation
"""

from __future__ import print_function

import copy
import os.path

from sisyphus import *
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.corpus.filter import FilterSegmentsByListJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.hdf import ReturnnDumpHDFJob
from sisyphus.delayed_ops import DelayedFunction

# sisyphus related
gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0]

# Datasets for wav2vec2
def get_returnn_datasets():   
    datasets = {"train": {}, "cv": {}}
    
    work_dir = "/u/dle/VietMed_05-09-2023/data_dump/"

    files = dict(
        config="/u/dle/VietMed_05-09-2023/data_dump/sprint.train.config2",
        # MFCC is used only for passing the stupid HDF setup
        features="/u/dle/VietMed_05-09-2023/data_dump/mfcc_vietmed_train_v1.cache.bundle",
        alignment="/u/dle/VietMed_05-09-2023/data_dump/alignment_sat.cache.bundle",
        corpus="/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_train_v1.xml.gz",
        lexicon="/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz",
        allophones="/u/dle/VietMed_05-09-2023/data_dump/allophones",
        cart="/u/dle/VietMed_05-09-2023/data_dump/cart.tree.xml.gz",
    )
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
        files[k] = Path(v, cached=True)
    segments = SegmentCorpusJob(files["corpus"], 1).out_single_segment_files

    segments_black_list = sorted([
        #"vietmed_corpus_train_v1_merged/VietMed_011_b/VietMed_011_b/VietMed_011_b_423",
    ])        

    segments = FilterSegmentsByListJob(
        segments, segments_black_list
    ).out_single_segment_files[1]

    segment_job = ShuffleAndSplitSegmentsJob(
        segment_file=segments,
        split={"segments_train": 0.97, "segments_cv": 0.03},
        shuffle=True,
        #shuffle=False,
    )
    files.update(segment_job.out_segments)

    # create ogg zip dataset
    ogg_zip_dict = {}
    for name in datasets:
        ogg_zip_job = BlissToOggZipJob(
            files["corpus"],
            segments=files[f"segments_{name}"],
            #rasr_cache=files["features"],
            raw_sample_rate=8000,
            feat_sample_rate=100,        
        )
        ogg_zip_job.rqmt = {"cpu": 1, "mem": 12, "time": 168}
        ogg_zip_job.add_alias(os.path.join("datasets", f"{name}_ogg_zip_job"))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip
        tk.register_output(
            os.path.join("datasets", f"{name}_ogg.zip"), ogg_zip_dict[name]
        )
        datasets[name]["ogg"] = ogg_zip_job.out_ogg_zip

    # create hdf dataset
    def get_sprint_dataset(corpus):
        assert corpus in ["train", "cv"]
        epoch_split = {"train": 1, "cv": 1}  # has to be 1 to dump the whole dataset

        estimated_num_seqs = {"train": 97670, "cv": 2999}[corpus]
        num_outputs = 4501
        corpus_order = {
            "train": "--*.corpus.segment-order-shuffle=true",
            "cv": "--*.corpus.segment-order-sort-by-time-length=true",
        }

        args = [
            "--config=" + files["config"].get(),
            "--*.corpus.file=" + files["corpus"].get(),
            "--*.corpus.segments.file=" + files[f"segments_{corpus}"].get(),
            corpus_order[corpus],
            "--*.state-tying.type=cart",
            "--*.state-tying.file=" + files["cart"].get(),
            "--*.trainer-output-dimension=%i" % num_outputs,
            "--*.lexicon.file=" + files["lexicon"].get(),
            "--*.allophones.add-from-file=" + files["allophones"].get(),
            "--*.alignment-cache-path=" + files["alignment"].get(),
            "--*.feature-cache-path=" + files["features"].get(),
            "--*.log-channel.file=log/crnn.sprint.%s.xml" % corpus,
            "--*.log-channel.compressed=false",
            "--*.window-size=1",
            "--*.trainer-output-dimension=%i" % num_outputs,
        ]
        nn_exe = "/work/asr4/vieting/programs/rasr/20230704/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard"
        return {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": nn_exe,
            "sprintConfigStr": args,
            "partitionEpoch": epoch_split[corpus],
            "estimated_num_seqs": estimated_num_seqs,
        }

    for name in datasets:
        data = get_sprint_dataset(name)
        hdf_job = ReturnnDumpHDFJob(str(data), mem=32)
        hdf_job.add_alias(os.path.join("datasets", f"{name}_hdf_job"))
        tk.register_output(
            os.path.join("datasets", f"{name}_alignments.hdf"), hdf_job.out_hdf
        )
        datasets[name]["hdf"] = hdf_job.out_hdf

    datasets["train"]["meta"] = {
        "class": "MetaDataset",
        "context_window": {"classes": 1, "data": 500},
        "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
        "datasets": {
            "hdf": {
                "class": "HDFDataset",
                "files": [datasets["train"]["hdf"]],
            },
            "ogg": {
                "audio": {"features": "raw", "sample_rate": 8000},
                "class": "OggZipDataset",
                "partition_epoch": 6,
                "path": datasets["train"]["ogg"],
                "seq_ordering": "laplace:226",
                "targets": None,
            },
        },
        "seq_order_control_dataset": "ogg",
    }
    datasets["cv"]["meta"] = {
        "class": "MetaDataset",
        "context_window": {"classes": 1, "data": 500},
        "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
        "datasets": {
            "hdf": {
                "class": "HDFDataset",
                "files": [datasets["cv"]["hdf"]],
            },
            "ogg": {
                "audio": {"features": "raw", "sample_rate": 8000},
                "class": "OggZipDataset",
                "fixed_random_seed": 1,
                "path": datasets["cv"]["ogg"],
                "seq_ordering": "sorted_reverse",
                "targets": None,
            },
        },
        "seq_order_control_dataset": "ogg",
    }

    return datasets
    
    
# Datasets for Gammatone supervised-only baselines
def get_returnn_datasets_gt_supervised():   
    datasets = {"train": {}, "cv": {}}
    
    work_dir = "/u/dle/VietMed_05-09-2023/data_dump/"

    files = dict(
        config="/u/dle/VietMed_05-09-2023/data_dump/sprint.train.config2",
        # Gammatone
        features="/u/dle/VietMed_05-09-2023/data_dump/gt_vietmed_train_v1.cache.bundle",
        alignment="/u/dle/VietMed_05-09-2023/data_dump/alignment_sat.cache.bundle",
        corpus="/u/dle/VietMed_05-09-2023/data_dump/vietmed_corpus_train_v1.xml.gz",
        lexicon="/u/dle/VietMed_05-09-2023/data_dump/g2p_augmented_oov.lexicon.gz",
        allophones="/u/dle/VietMed_05-09-2023/data_dump/allophones",
        cart="/u/dle/VietMed_05-09-2023/data_dump/cart.tree.xml.gz",
    )
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
        files[k] = Path(v, cached=True)
    segments = SegmentCorpusJob(files["corpus"], 1).out_single_segment_files

    segments_black_list = sorted([
        #"vietmed_corpus_train_v1_merged/VietMed_011_b/VietMed_011_b/VietMed_011_b_423",
    ])        

    segments = FilterSegmentsByListJob(
        segments, segments_black_list
    ).out_single_segment_files[1]

    segment_job = ShuffleAndSplitSegmentsJob(
        segment_file=segments,
        split={"segments_train": 0.97, "segments_cv": 0.03},
        shuffle=True,
        #shuffle=False,
    )
    files.update(segment_job.out_segments)

    # create ogg zip dataset
    ogg_zip_dict = {}
    for name in datasets:
        ogg_zip_job = BlissToOggZipJob(
            files["corpus"],
            segments=files[f"segments_{name}"],
            #rasr_cache=files["features"],
            raw_sample_rate=8000,
            feat_sample_rate=100,        
        )
        ogg_zip_job.rqmt = {"cpu": 1, "mem": 12, "time": 168}
        ogg_zip_job.add_alias(os.path.join("datasets", f"{name}_ogg_zip_job"))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip
        tk.register_output(
            os.path.join("datasets", f"{name}_ogg.zip"), ogg_zip_dict[name]
        )
        datasets[name]["ogg"] = ogg_zip_job.out_ogg_zip

    # create hdf dataset
    def get_sprint_dataset(corpus):
        assert corpus in ["train", "cv"]
        epoch_split = {"train": 1, "cv": 1}  # has to be 1 to dump the whole dataset

        estimated_num_seqs = {"train": 97670, "cv": 2999}[corpus]
        num_outputs = 4501
        corpus_order = {
            "train": "--*.corpus.segment-order-shuffle=true",
            "cv": "--*.corpus.segment-order-sort-by-time-length=true",
        }

        args = [
            "--config=" + files["config"].get(),
            "--*.corpus.file=" + files["corpus"].get(),
            "--*.corpus.segments.file=" + files[f"segments_{corpus}"].get(),
            corpus_order[corpus],
            "--*.state-tying.type=cart",
            "--*.state-tying.file=" + files["cart"].get(),
            "--*.trainer-output-dimension=%i" % num_outputs,
            "--*.lexicon.file=" + files["lexicon"].get(),
            "--*.allophones.add-from-file=" + files["allophones"].get(),
            "--*.alignment-cache-path=" + files["alignment"].get(),
            "--*.feature-cache-path=" + files["features"].get(),
            "--*.log-channel.file=log/crnn.sprint.%s.xml" % corpus,
            "--*.log-channel.compressed=false",
            "--*.window-size=1",
            "--*.trainer-output-dimension=%i" % num_outputs,
        ]
        nn_exe = "/work/asr4/vieting/programs/rasr/20230704/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard"
        return {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": nn_exe,
            "sprintConfigStr": args,
            "partitionEpoch": epoch_split[corpus],
            "estimated_num_seqs": estimated_num_seqs,
        }

    for name in datasets:
        data = get_sprint_dataset(name)
        hdf_job = ReturnnDumpHDFJob(str(data), mem=32)
        hdf_job.add_alias(os.path.join("datasets", f"{name}_hdf_job"))
        tk.register_output(
            os.path.join("datasets", f"{name}_alignments.hdf"), hdf_job.out_hdf
        )
        datasets[name]["hdf"] = hdf_job.out_hdf

    datasets["train"]["meta"] = {
        "class": "MetaDataset",
        "context_window": {"classes": 1, "data": 500},
        "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
        "datasets": {
            "hdf": {
                "class": "HDFDataset",
                "files": [datasets["train"]["hdf"]],
            },
            "ogg": {
                "audio": {"features": "raw", "sample_rate": 8000},
                "class": "OggZipDataset",
                "partition_epoch": 6,
                "path": datasets["train"]["ogg"],
                "seq_ordering": "laplace:226",
                "targets": None,
            },
        },
        "seq_order_control_dataset": "ogg",
    }
    datasets["cv"]["meta"] = {
        "class": "MetaDataset",
        "context_window": {"classes": 1, "data": 500},
        "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
        "datasets": {
            "hdf": {
                "class": "HDFDataset",
                "files": [datasets["cv"]["hdf"]],
            },
            "ogg": {
                "audio": {"features": "raw", "sample_rate": 8000},
                "class": "OggZipDataset",
                "fixed_random_seed": 1,
                "path": datasets["cv"]["ogg"],
                "seq_ordering": "sorted_reverse",
                "targets": None,
            },
        },
        "seq_order_control_dataset": "ogg",
    }

    return datasets
    
    
# Run all
#get_returnn_datasets()
get_returnn_datasets_gt_supervised()
    
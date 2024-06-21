"""SLUE evaluation script for NER tasks
Adapted from SLUE (Suwon et al., 2022) with github link:
https://github.com/asappresearch/slue-toolkit/blob/main/slue_toolkit/eval/eval_utils_ner.py"""

from collections import defaultdict
import numpy as np


def get_slue_format(
    words: list[str], tags: list[str], use_dummy: bool = False
) -> list[tuple[str, str, int]]:
    """Convert words and tags to SLUE format.

    Args:
        words: list of words
        tags: list of tags
        use_dummy: bool, whether to use dummy word for dummy tags.
                   Use dummy to ignore words and only focus on the existence of tags

    Returns:
        slue_tags: list of tuples of (tag, word, count)
    """

    # Remove B-, I-, O- prefixes from tags
    tags = [tag.split("-")[-1] for tag in tags]

    slue_tags = []
    for i, word in enumerate(words):
        if use_dummy:
            word = "dummy"
        count = 0
        tag = tags[i]
        # Ignore "dum" and "0" tags
        if tag not in ["dum", "0"]:
            tag_tuple = (tag, word, count)
            while tag_tuple in slue_tags:
                count += 1
                tag_tuple = (tag, word, count)
            slue_tags.append(tag_tuple)
    return slue_tags


def get_ner_scores(
    all_gt: list[list[tuple]], all_predictions: list[list[tuple]]
) -> dict:
    """Evaluate per-label and overall (micro and macro) metrics of precision, recall, and fscore.

    Args:
        all_gt: List of list of ground truth tuples (label, phrase, identifier)
        all_predictions: List of list of predicted tuples (label, phrase, identifier)

    Returns:
        Dictionary of metrics
    """
    metrics: dict = {}
    stats = get_ner_stats(all_gt, all_predictions)

    num_correct, num_gt, num_pred = 0, 0, 0
    prec_lst, recall_lst, fscore_lst = [], [], []

    for tag_name, tag_stats in stats.items():
        precision, recall, fscore = get_metrics(
            np.sum(tag_stats["tp"]),
            np.sum(tag_stats["gt_cnt"]),
            np.sum(tag_stats["pred_cnt"]),
        )
        metrics.setdefault(tag_name, {})
        metrics[tag_name]["precision"] = precision
        metrics[tag_name]["recall"] = recall
        metrics[tag_name]["fscore"] = fscore

        num_correct += np.sum(tag_stats["tp"])
        num_pred += np.sum(tag_stats["pred_cnt"])
        num_gt += np.sum(tag_stats["gt_cnt"])

        prec_lst.append(precision)
        recall_lst.append(recall)
        fscore_lst.append(fscore)

    # Calculate overall micro metrics
    precision, recall, fscore = get_metrics(num_correct, num_gt, num_pred)
    metrics["overall_micro"] = {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }

    # Calculate overall macro metrics
    metrics["overall_macro"] = {
        "precision": np.mean(prec_lst),
        "recall": np.mean(recall_lst),
        "fscore": np.mean(fscore_lst),
    }

    return metrics


def get_ner_stats(
    all_gt: list[list[tuple]], all_predictions: list[list[tuple]]
) -> dict:
    """Compute true positives, ground truth counts, and prediction counts for each tag.

    Args:
        all_gt: List of list of ground truth tuples (label, phrase, identifier)
        all_predictions: List of list of predicted tuples (label, phrase, identifier)

    Returns:
        Dictionary with statistics for each tag
    """
    stats: dict = {}
    for gt, pred in zip(all_gt, all_predictions):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)

        for type_name, entity_info1, entity_info2 in gt:
            entities_true[type_name].add((entity_info1, entity_info2))
        for type_name, entity_info1, entity_info2 in pred:
            entities_pred[type_name].add((entity_info1, entity_info2))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        for tag_name in target_names:
            stats.setdefault(tag_name, {"tp": [], "gt_cnt": [], "pred_cnt": []})
            entities_true_type = entities_true.get(tag_name, set())
            entities_pred_type = entities_pred.get(tag_name, set())

            stats[tag_name]["tp"].append(len(entities_true_type & entities_pred_type))
            stats[tag_name]["pred_cnt"].append(len(entities_pred_type))
            stats[tag_name]["gt_cnt"].append(len(entities_true_type))

    return stats


def safe_divide(numerator, denominator):
    """Divide two arrays, avoiding division by zero.
    Args:
        numerator: Numerator array
        denominator: Denominator array
    Returns:
        Array of division results"""

    numerator = np.array(numerator)
    denominator = np.array(denominator)
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    return numerator / denominator


def ner_error_analysis(all_gt, all_predictions, gt_text):
    """Generate error analysis examples for NER tasks.
    Args:
        all_gt: List of list of ground truth tuples (label, phrase, identifier)
        all_predictions: List of list of predicted tuples (label, phrase, identifier)
        gt_text: List of ground truth text
    Returns:
        Dictionary of error analysis examples"""
    analysis_examples_dct = {}
    analysis_examples_dct["all"] = []
    for idx, text in enumerate(gt_text):
        if isinstance(text, list):
            text = " ".join(text)
        gt = all_gt[idx]
        pred = all_predictions[idx]
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, entity_info1, entity_info2 in gt:
            entities_true[type_name].add((entity_info1, entity_info2))
        for type_name, entity_info1, entity_info2 in pred:
            entities_pred[type_name].add((entity_info1, entity_info2))
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        analysis_examples_dct["all"].append("\t".join([text, str(gt), str(pred)]))
        for tag_name in target_names:
            _ = analysis_examples_dct.setdefault(tag_name, [])
            new_gt = [(item1, item2) for item1, item2, _ in gt]
            new_pred = [(item1, item2) for item1, item2, _ in pred]
            analysis_examples_dct[tag_name].append(
                "\t".join([text, str(new_gt), str(new_pred)])
            )

    return analysis_examples_dct


def get_metrics(num_correct, num_gt, num_pred):
    """Get precision, recall, and fscore metrics.
    Args:
        num_correct: Number of correct predictions
        num_gt: Number of ground truth
        num_pred: Number of predictions
    Returns:
        Tuple of precision, recall, and fscore"""
    precision = safe_divide([num_correct], [num_pred])
    recall = safe_divide([num_correct], [num_gt])
    fscore = safe_divide([2 * precision * recall], [(precision + recall)])
    return precision[0], recall[0], fscore[0][0]

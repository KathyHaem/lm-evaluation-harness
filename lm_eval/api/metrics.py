import logging
import math
import os
import random
from collections.abc import Iterable
from typing import List

import evaluate as hf_evaluate
import numpy as np
import sacrebleu
import sklearn.metrics
import wget
import fasttext
from langcodes import Language, standardize_tag

from lm_eval.api.registry import register_aggregation, register_metric


eval_logger = logging.getLogger("lm-eval")
LANG_ID_MODEL: fasttext.FastText = None


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    # print(preds)
    return sklearn.metrics.matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("ngram_div")
def ngram_div(sequences, n=3):
    """Returns the mean of the fraction of unique k-grams for k in {1,...,n}.
    i.e., a list of means.
    """
    preds = list(zip(*sequences))[1]
    divs = np.array([dist_k(preds, k) for k in range(1, n + 1)])
    return divs.mean(axis=0).tolist()[2]  # am reducing it to one number, just taking trigram diversity


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    gold = list(gold)
    gold_one_hot = np.eye(np.max(gold) + 1)[gold]
    predictions = list(zip(*items))[1]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


exact_match = hf_evaluate.load("exact_match")


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match.compute(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="trigram_div",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ngram_div"
)
def trigram_div_fn(items):  # this is a passthrough function
    return items


@register_metric(
    metric="lang_id",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def lang_id_fn(item):
    if not LANG_ID_MODEL:
        load_lang_id_model()
    ref = item[0]
    pred = item[1]
    pred_id = LANG_ID_MODEL.predict(text=pred)
    pred_id = pred_id[0][0].split("__")[2]  # actual output format: (('__label__eng_Latn',), array([1.00001001]))
    ref_id = LANG_ID_MODEL.predict(text=ref)
    ref_id = ref_id[0][0].split("__")[2]  # actual output format: (('__label__eng_Latn',), array([1.00001001]))
    return ref_id == pred_id


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


#### from https://github.com/gianwiher/decoding-NLG ####

def calc_repetitions(sequences):
    """
    code adapted from https://github.com/ari-holtzman/degen/blob/master/metrics/repetition.py

    returns a list of dicts which contains detailed fields on how each
    example is repeating itself, specifically the phrase the generation is repeating
    and how many times it is repeated.
    """

    objs = []
    max_n = 90
    n_repeated_examples = 0

    for gen in sequences:
        obj = {}
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while (
                len(rev_gen[n * n_repeat : n * (n_repeat + 1)]) == n
                and rev_gen[n * n_repeat : n * (n_repeat + 1)] == rev_gen[:n]
            ):
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])
        if last_n_repeats[max_repeated_n] > 1 and (
            max_repeated_n + 1 >= 3 or last_n_repeats[max_repeated_n] > 50
        ):
            obj["repetition"] = {
                "repeated_phrase": list(reversed(rev_gen[: max_repeated_n + 1])),
                "repeated_times": last_n_repeats[max_repeated_n],
                "repeated_phrase_length": max_repeated_n + 1,
            }
            n_repeated_examples += 1
        else:
            obj["repetition"] = None

        objs.append(obj)

    return objs


def get_k_grams(sequence, k):
    """Returns the k-grams of the input sequence"""

    grams = []
    if len(sequence) < k:
        print(f"Warning: input sequence len is {len(sequence)} but k is {k}")
        return np.nan

    for i in range(len(sequence) - k + 1):
        grams.append(tuple(sequence[i : i + k]))

    return grams


def dist_k(preds, k):
    """
    Number of unique k-grams divided by the number of tokens
    :param sequences: a list of sequences of tokens
    :param k: size of k-gram
    :return: list of fractions unique/total k-grams in the sequence
    """
    res = []
    for sequence in preds:
        kgrams = get_k_grams(sequence, k)
        if kgrams is np.nan:
            res.append(np.nan)
            continue
        unique = len(set(kgrams))
        total = len(sequence) - k + 1
        res.append(unique / total if total != 0 else np.nan)
    return res


#### up to here ####

def load_lang_id_model():
    # lid_model = "../../lid201-model.ftz"  # wikimedia one
    lid_model = "../../model.bin"  # glotlid one, seems better
    if not os.path.isfile(lid_model):
        # model_url = "https://data.statmt.org/lid/lid201-model.ftz"  # wikimedia one
        model_url = "https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin"  # glotlid one
        wget.download(model_url, out="../../")
    global LANG_ID_MODEL
    LANG_ID_MODEL = fasttext.load_model(lid_model)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert (
        metrics is not None
    ), "Need to pass a list of each subtask's metric for this stderr aggregation"
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)

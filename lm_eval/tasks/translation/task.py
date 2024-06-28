"""
NOTE: This file implements translation tasks using datasets from WMT conferences,
provided by sacrebleu. Traditionally they are evaluated with BLEU scores. TER
and CHRF are other options.

This is most of the old implementation file.

We defer citations and descriptions of the many translations tasks used
here to the SacreBLEU repo from which we've obtained the datasets:
https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py

Homepage: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py
"""
import os
from typing import List

import pycountry
from sacrebleu import sacrebleu

from lm_eval.api.task import ConfigurableTask

try:
    import nagisa

    HAS_NAGISA = True
except ImportError:
    HAS_NAGISA = False

try:
    import jieba

    jieba.dt.tmp_dir = os.path.join(os.getcwd(), "jieba_tmp")
    os.makedirs(jieba.dt.tmp_dir, exist_ok=True)

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

_CITATION = """
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""

sacrebleu_datasets = sacrebleu.DATASETS


def create_tasks_from_benchmarks(benchmark_dict):
    """Creates a dictionary of tasks from a dict
    :param benchmark_dict: { dataset: [lang_pair, ...], }
    :return: {task_name: task}
        e.g. {wmt14-fr-en: Task, wmt16-de-en: Task}
    """

    def version_of(dataset, language_pair):
        if language_pair[-2:] in ["zh", "ja"]:
            return 1  # changed to use jieba/nagisa
        return 0

    return {
        f"{dataset}-{language_pair}": create_translation_task(
            dataset, language_pair, version_of(dataset, language_pair)
        )
        for dataset, language_pairs in benchmark_dict.items()
        for language_pair in language_pairs
    }


########################################
# Language Specifics
########################################


def zh_split(zh_text: List[str]) -> List[str]:
    """Chinese splitting"""
    if not HAS_JIEBA:
        raise ImportError(
            "Chinese text splitting requires the `jieba` package. "
            "Please install it with:\npip install jieba"
        )

    return [" ".join(jieba.cut(txt.strip())) for txt in zh_text]


def ja_split(ja_text: List[str]) -> List[str]:
    """Japanese splitting"""
    if not HAS_NAGISA:
        raise ImportError(
            "Japanese text splitting requires the `nagisa` package. "
            "Please install it with:\npip install nagisa"
        )

    return [" ".join(nagisa.tagging(txt.strip()).words) for txt in ja_text]


NO_SPACE_LANG = {"zh": zh_split, "ja": ja_split}


########################################
# Tasks
########################################


def create_translation_task(dataset, language_pair, version=0):
    class TranslationTask(GeneralTranslationTask):
        VERSION = version

        def __init__(self):
            super().__init__(config={"dataset_path": dataset, "dataset_name": language_pair})

    return TranslationTask


class GeneralTranslationTask(ConfigurableTask):
    VERSION = 0

    # e.g. ("wmt14", "fr-en")
    def __init__(self, config):
        config.pop("class")
        self.sacrebleu_dataset = config["dataset_path"]
        self.sacrebleu_language_pair = config["dataset_name"]
        self.src_file = self.ref_file = self.src_data = self.ref_data = None
        self.lang_id_model = None

        super().__init__(config=config)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # This caches in the users home dir automatically, returns a list in newer versions
        task_files = sacrebleu.download_test_set(
            self.sacrebleu_dataset, self.sacrebleu_language_pair
        )
        print(task_files)
        self.src_file = task_files[0]
        self.ref_file = task_files[1]
        self.src_data, self.ref_data = [
            [line.rstrip() for line in sacrebleu.smart_open(file)]
            for file in (self.src_file, self.ref_file)
        ]

    def has_training_docs(self):
        """Whether the task has a training set"""
        # TODO In the future we could be more discerning. Some more recent tests have train and dev sets
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return [
            {"src": src, "ref": ref} for src, ref in zip(self.src_data, self.ref_data)
        ]

    def doc_to_text(self, doc):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{src_lang} phrase: " + doc["src"] + f"\n{tar_lang} phrase:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["src"]

    def doc_to_target(self, doc):
        # This shows a single target, though there may be multiple targets in a lang test
        return " " + doc["ref"] if isinstance(doc["ref"], str) else doc["ref"][0]

    def __str__(self):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{self.sacrebleu_dataset.upper()} {src_lang} to {tar_lang} Task"


########################################
# Util
########################################


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name

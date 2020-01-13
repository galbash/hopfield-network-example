"""
main handler for mating network
"""
import logging
import sys
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict
import progressbar
import numpy
from fire import Fire

from mater.network import MatcherNetwork, NetworkParams
from mater.ranks import random_arrays

MAX_EPOCHS = 1000
MAX_NO_CHANGE_EPOCS = 20
TEST_COUNT = 150


@dataclass
class TestResult:
    valid: bool
    score: float
    result: numpy.ndarray


def run_test(ranks: numpy.ndarray, params: NetworkParams) -> TestResult:
    """
    Creates a network with the given parameters and attempts to solve
    for the given ranks
    :return: the results for the Test
    """
    network = MatcherNetwork(ranks, params)
    no_update_count = 0
    total_count = 0
    while no_update_count < MAX_NO_CHANGE_EPOCS and total_count < MAX_EPOCHS:
        was_updated = network.random_epoch()
        if was_updated:
            no_update_count = 0
        else:
            no_update_count += 1
        total_count += 1

    score = numpy.sum(network.ranks * network.neurons)
    valid_count = numpy.count_nonzero(network.neurons) == 10
    valid_rows = all(x == 1 for x in numpy.sum(network.neurons, axis=1))
    valid_cols = all(x == 1 for x in numpy.sum(network.neurons, axis=0))
    return TestResult(valid_cols and valid_rows and valid_count, score, network.neurons)


def run_for_params(
    ranks: numpy.ndarray, params: NetworkParams, iterations_count: int
) -> List[TestResult]:
    """
    runs many iterations for given ranks and returns all the results
    """
    test_results = []
    with progressbar.ProgressBar(max_value=iterations_count) as bar:
        for i in range(TEST_COUNT):
            result = run_test(ranks, params)
            test_results.append(result)
            bar.update(i)

    return test_results


@dataclass
class TestSummary:
    """
    Summary for a batch test for given parameters
    """

    params: NetworkParams
    convergence_rate: float
    top_score: float = None
    median_score: float = None

    @property
    def csv(self) -> Dict[str, object]:
        """
        :return: summary as csv-ready dict
        """
        self_dict = asdict(self)
        params = self_dict.pop("params")
        return {**self_dict, **params}


def run_for_ranks(
    ranks: numpy.ndarray, params: List[NetworkParams], logger: logging.Logger
) -> List[TestSummary]:
    """
    runs a batch of tests for each of the given parameters and report
    statistics to the given logger
    """
    logger.info("testing ranks:\n %s", ranks)
    summaries = []
    for param in params:
        logger.info("using params: %s", param)
        results = run_for_params(ranks, param, TEST_COUNT)
        valid_results = sorted(
            [result for result in results if result.valid],
            key=lambda result: result.score,
        )
        if valid_results:
            summary = TestSummary(
                param,
                len(valid_results) / len(results),
                valid_results[0].score,
                valid_results[len(valid_results) // 2].score,
            )
        else:
            summary = TestSummary(param, 0)

        summaries.append(summary)
        logger.info("convergence rate: %.2f%%:", (summary.convergence_rate * 100))
        if summary.top_score is not None:
            logger.info("top result:\n %s", valid_results[0].result)
            logger.info("top result score: %f", summary.top_score)
            logger.info("median result (of converged): %f", summary.median_score)
        else:
            logging.info("no runs converged")

    return summaries


def setup_logging() -> logging.Logger:
    """
    Sets up the test logger
    """
    logger = logging.getLogger("mating-network")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("test.log", mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


TESTED_PARAMS = [
    NetworkParams(100, 100, 90, 20, 1.1),
    NetworkParams(100, 100, 90, 30, 1.1),
    NetworkParams(100, 100, 90, 40, 1.1),
    NetworkParams(100, 100, 90, 45, 1.1),
    NetworkParams(100, 100, 90, 50, 1.1),
    NetworkParams(100, 100, 90, 20, 1),
    NetworkParams(100, 100, 90, 30, 1),
    NetworkParams(100, 100, 90, 40, 1),
    NetworkParams(100, 100, 90, 45, 1),
    NetworkParams(100, 100, 90, 50, 1),
]

FIELD_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "sigma",
    "alpha",
    "convergence_rate",
    "top_score",
    "median_score",
]


def save_result(summaries: List[TestSummary]) -> None:
    with open("report.csv", "wt") as f:
        writer = csv.DictWriter(f, FIELD_NAMES)
        writer.writeheader()
        writer.writerows([summary.csv for summary in summaries])


def main():
    """
    main entry point
    """
    logger = setup_logging()
    ranks = random_arrays()
    summaries = run_for_ranks(ranks, TESTED_PARAMS, logger)
    save_result(summaries)


if __name__ == "__main__":
    Fire(main)

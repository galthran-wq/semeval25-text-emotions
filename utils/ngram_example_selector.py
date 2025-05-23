"""Select and order examples based on ngram overlap score (sentence_bleu score).

https://www.nltk.org/_modules/nltk/translate/bleu_score.html
https://aclanthology.org/P02-1040.pdf
"""

from typing import Dict, List, Optional, Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator


def ngram_overlap_score(source: List[str], example: List[str]) -> float:
    """Compute ngram overlap score of source and example as sentence_bleu score
    from NLTK package.

    Use sentence_bleu with method1 smoothing function and auto reweighting.
    Return float value between 0.0 and 1.0 inclusive.
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
    from nltk.translate.bleu_score import (
        SmoothingFunction,  # type: ignore
        sentence_bleu,
    )

    hypotheses = source[0].split()
    references = [s.split() for s in example]

    return float(
        sentence_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1,
            auto_reweigh=True,
        )
    )



class NGramOverlapExampleSelector(BaseExampleSelector, BaseModel):
    """Select and order examples based on ngram overlap score (sentence_bleu score
    from NLTK package).

    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """

    examples: List[Any]
    """A list of the examples that the prompt template expects."""

    threshold: float = -1.0
    """Threshold at which algorithm stops. Set to -1.0 by default.

    For negative threshold:
    select_examples sorts examples by ngram_overlap_score, but excludes none.
    For threshold greater than 1.0:
    select_examples excludes all examples, and returns an empty list.
    For threshold equal to 0.0:
    select_examples sorts examples by ngram_overlap_score,
    and excludes examples with no ngram overlap with input.
    """

    @root_validator(pre=True)
    def check_dependencies(cls, values: Dict) -> Dict:
        """Check that valid dependencies exist."""
        try:
            from nltk.translate.bleu_score import (  # noqa: F401
                SmoothingFunction,
                sentence_bleu,
            )
        except ImportError as e:
            raise ImportError(
                "Not all the correct dependencies for this ExampleSelect exist."
                "Please install nltk with `pip install nltk`."
            ) from e

        return values

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[Document]:
        """Return list of examples sorted by ngram_overlap_score with input.

        Descending order.
        Excludes any examples with ngram_overlap_score less than or equal to threshold.
        """
        inputs = [input_variables["text"]]
        examples = []
        k = len(self.examples)
        score = [0.0] * k

        for i in range(k):
            score[i] = ngram_overlap_score(
                inputs, [self.examples[i].page_content]
            )

        while True:
            arg_max = np.argmax(score)
            if (score[arg_max] < self.threshold) or abs(
                score[arg_max] - self.threshold
            ) < 1e-9:
                break

            examples.append(self.examples[arg_max])
            score[arg_max] = self.threshold - 1.0

        return examples


class NGramOverlapKExampleSelector(NGramOverlapExampleSelector):
    k: Optional[int] = None

    def format_docs(self, docs):
        return [
            {
                "text": doc.page_content, 
                "result": doc.metadata["result"]
            }
            for doc in docs
        ]

    def select_examples(self, input_variables):
        examples = super().select_examples(input_variables)
        return self.format_docs(examples[:self.k])

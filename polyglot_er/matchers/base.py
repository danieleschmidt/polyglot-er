"""
Abstract base class for cross-lingual matchers.

All matchers in the polyglot-er pipeline implement this interface so they
can be composed into the 5-tier cascade.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchResult:
    """
    Result of a cross-lingual entity match attempt.

    Attributes:
        is_match: Whether the two entities are considered the same
        score: Similarity score in [0, 1]
        tier: Which cascade tier produced this result (0–4, or None if undecided)
        method: Human-readable name of the matching method
        details: Optional extra information for debugging
    """

    is_match: bool
    score: float
    tier: Optional[int] = None
    method: str = ""
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MatchResult(match={self.is_match}, score={self.score:.3f}, "
            f"tier={self.tier}, method={self.method!r})"
        )


class CrossLingualMatcher(ABC):
    """
    Abstract base for all cross-lingual entity matchers.

    Subclasses implement ``score`` to return a float in [0, 1] and
    ``match`` to return a MatchResult.
    """

    #: Default similarity threshold for a positive match
    DEFAULT_THRESHOLD: float = 0.75

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

    @abstractmethod
    def score(self, name_a: str, name_b: str, lang_a: str = "", lang_b: str = "") -> float:
        """
        Compute a similarity score between two entity name strings.

        Args:
            name_a: First entity name
            name_b: Second entity name
            lang_a: BCP-47 language code for name_a (optional hint)
            lang_b: BCP-47 language code for name_b (optional hint)

        Returns:
            Float in [0, 1] where 1.0 means identical/perfect match
        """
        ...

    def match(
        self,
        name_a: str,
        name_b: str,
        lang_a: str = "",
        lang_b: str = "",
        tier: Optional[int] = None,
    ) -> MatchResult:
        """
        Determine whether two entity names refer to the same entity.

        Args:
            name_a: First entity name
            name_b: Second entity name
            lang_a: BCP-47 language code for name_a
            lang_b: BCP-47 language code for name_b
            tier: Cascade tier number (for provenance)

        Returns:
            MatchResult
        """
        s = self.score(name_a, name_b, lang_a, lang_b)
        return MatchResult(
            is_match=s >= self.threshold,
            score=s,
            tier=tier,
            method=self.__class__.__name__,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"

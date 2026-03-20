"""Datasets: synthetic entity pairs and Wikidata stub loader."""

from .synthetic import generate_synthetic_entities, load_synthetic_entities
from .wikidata import WikidataLoader

__all__ = ["generate_synthetic_entities", "load_synthetic_entities", "WikidataLoader"]

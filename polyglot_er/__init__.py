"""
polyglot_er — Cross-Lingual Entity Resolution Pipeline

Resolves the same real-world entity appearing with different name forms across
multiple languages using a 5-tier cascade: normalization → type-check → 
same-script fuzzy → cross-script phonetic → multilingual embeddings.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .resolver import CrossLingualResolver

__all__ = ["CrossLingualResolver", "__version__"]

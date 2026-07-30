"""Language — statistical natural-language understanding without any APIs."""

from .text import LanguageCortex, tokenize, sentence_split

__all__ = ["LanguageCortex", "tokenize", "sentence_split"]

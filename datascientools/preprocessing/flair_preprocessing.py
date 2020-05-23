from typing import List

import torch
from janome.tokenizer import Tokenizer
import flair
from flair.data import Token, Sentence
from flair.embeddings import (
    WordEmbeddings,
    TokenEmbeddings,
    FlairEmbeddings,
)
from flair.embeddings import (
    DocumentEmbeddings,
    DocumentPoolEmbeddings,
)

tokenizer = Tokenizer()

def janome_tokenizer(text: str) -> List[str]:
    return [token.surface for token in tokenizer.tokenize(text)]

ja_wiki_word_embeddings: TokenEmbeddings = WordEmbeddings("ja-wiki")  # or ja-crawl

document_embeddings: DocumentEmbeddings = DocumentPoolEmbeddings(
    embeddings=[ja_wiki_word_embeddings],
    pooling="mean",
)

def get_text_embeddings(text: str) -> torch.Tensor:
    sentence = Sentence(text, use_tokenizer=janome_tokenizer)
    document_embeddings.embed(sentence)
    return sentence.embedding


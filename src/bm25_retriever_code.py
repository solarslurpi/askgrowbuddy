import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, cast, Union
from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import BaseDocumentStore

import bm25s
import Stemmer


logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}

DEFAULT_PERSIST_FILENAME = "retriever.json"


class BM25Retriever(BaseRetriever):
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        stemmer (Stemmer.Stemmer, optional):
            The stemmer to use. Defaults to an english stemmer.
        language (str, optional):
            The language to use for stopword removal. Defaults to "en".
        existing_bm25 (bm25s.BM25, optional):
            An existing BM25 object to use. If not provided, nodes must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25s.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        self.stemmer = stemmer or Stemmer.Stemmer("english")
        self.similarity_top_k = similarity_top_k
        self._verbose = verbose

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            def custom_node_to_metadata_dict(node: BaseNode) -> Dict[str, Any]:
                metadata = node.metadata.copy() if node.metadata else {}
                metadata["text"] = node.get_content()
                metadata["id"] = node.node_id
                return metadata

            self.corpus = [custom_node_to_metadata_dict(node) for node in nodes]

            corpus_tokens = bm25s.tokenize(
                [node.get_content() for node in nodes],
                stopwords=language,
                stemmer=self.stemmer,
                show_progress=verbose,
            )
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens, show_progress=verbose)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        # deprecated
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Retriever":
        if tokenizer is not None:
            logger.warning(
                "The tokenizer parameter is deprecated and will be removed in a future release. "
                "Use a stemmer from PyStemmer instead."
            )

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            stemmer=stemmer,
            language=language,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        return {
            DEFAULT_PERSIST_ARGS[key]: getattr(self, key)
            for key in DEFAULT_PERSIST_ARGS
            if hasattr(self, key)
        }

    def persist(self, path: str, **kwargs: Any) -> None:
        """Persist the retriever to a directory."""
        self.bm25.save(path, corpus=self.corpus, **kwargs)
        with open(os.path.join(path, DEFAULT_PERSIST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self.get_persist_args(), f, indent=2)

    @classmethod
    def from_persist_dir(cls, path: str, **kwargs: Any) -> "BM25Retriever":
        """Load the retriever from a directory."""
        bm25 = bm25s.BM25.load(path, load_corpus=True)
        with open(os.path.join(path, DEFAULT_PERSIST_FILENAME), encoding='utf-8') as f:
            retriever_data = json.load(f)
        # Update retriever data with any provided kwargs - like similarity_top_k
        retriever_data.update(kwargs)
        return cls(existing_bm25=bm25, **retriever_data)

    def _retrieve(self, query_bundle: QueryBundle) -> List[Document]:
        query = query_bundle.query_str
        tokenized_query = bm25s.tokenize(
            query, stemmer=self.stemmer, show_progress=self._verbose
        )
        indexes, scores = self.bm25.retrieve(
            tokenized_query, k=self.similarity_top_k, show_progress=self._verbose
        )

        nodes_with_scores: List[NodeWithScore] = []
        for idx, score in zip(indexes[0], scores[0]):
            # idx is already a dict containing the document information
            text_node = TextNode(
                text=idx["text"],
                node_id=idx["id"],
                metadata={k: v for k, v in idx.items() if k not in ["text", "id"]}
            )
            # Create a NodeWithScore from the TextNode
            node_with_score = NodeWithScore(node=text_node, score=score)
            nodes_with_scores.append(node_with_score)

        return nodes_with_scores

    def retrieve(self, str_or_query_bundle: Union[str, QueryBundle]) -> List[Document]:
        if isinstance(str_or_query_bundle, str):
            return self._retrieve(QueryBundle(query_str=str_or_query_bundle))
        documents = self._retrieve(str_or_query_bundle)
        return documents

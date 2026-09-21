__all__ = ["Retriever"]


def __getattr__(name):
    if name == "Retriever":
        from .passage_retriever import Retriever

        return Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from ...pandora_repo_defaults import ensure_pandora_repo_root_on_path

ensure_pandora_repo_root_on_path()

from qa_shared.prompts import (
    QUERY_GENERATION_BM25_SYSTEM_PROMPT,
    QUERY_GENERATION_CONTRIEVER_SYSTEM_PROMPT,
    QUERY_GENERATION_USER_PROMPT,
    gen_bm25_query_prompt,
    gen_contriever_query_prompt,
)

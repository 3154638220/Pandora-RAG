from ...pandora_repo_defaults import ensure_pandora_repo_root_on_path

ensure_pandora_repo_root_on_path()

from qa_shared.prompts import (
    ANSWER_STYLE_RULES as FINAL_ANSWER_GENERATION_SYSTEM_PROMPT,
    INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT,
    gen_final_answer_prompt,
    gen_intermediate_answer_prompt,
)

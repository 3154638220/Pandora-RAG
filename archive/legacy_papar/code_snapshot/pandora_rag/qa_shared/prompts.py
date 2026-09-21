"""Shared prompt templates for final answers and iterative retrieval traces."""

from typing import Iterable, Mapping, Optional


ANSWER_STYLE_RULES = """\
Given a multi-hop question with retrieved context, generate a final answer.

Rules:
1. Focus on answering the question.
2. Use the retrieved context as your primary evidence.
3. If the context is insufficient, provide your best guess.
4. Output only the answer, with no additional text.
5. Keep the answer concise and directly relevant to the question."""


ANSWER_PROMPT = """\
{rules}

Question: {question}

Retrieved context:
{context}

Answer:{answer_suffix}"""


def format_answer_prompt(question: str, context: str, answer: Optional[str] = None) -> str:
    answer_suffix = f" {answer}" if answer is not None else ""
    return ANSWER_PROMPT.format(
        rules=ANSWER_STYLE_RULES,
        question=question,
        context=context,
        answer_suffix=answer_suffix,
    )


def format_context_from_docs(docs: Iterable[Mapping[str, str]]) -> str:
    return "\n\n".join(f"{doc['title']}: {doc['text']}" for doc in docs)


INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT = """\
Given a question with its corresponding Wikipedia snippet, generate an answer using only the provided snippet.

Process:
- You receive:
  - Question: <a question>
  - Document: <Wikipedia snippet for the question>

- Your task:
    1. Focus on answering the question.
    2. Use only the provided Document as your evidence.
    3. Output only the answer, with no additional text.
    4. Keep the answer concise and directly relevant to the question.

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

#
## Input
Question: Where did Andre Bloc live when he died?
Document: Andre Bloc (Algiers, May 23, 1896 - New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
## Output
Andre Bloc died in New Delhi.

#
## Input
Question: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988.
## Output
Karin Thomas was born in Brusio.

#
## Input
Question: In which country is a spiral viaduct located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubunden, Switzerland.
## Output
The Brusio spiral viaduct is located in Switzerland.

#
## Input
Question: What is the least popular official language of Switzerland?
Document: Switzerland has four official languages: principally German (63.5% total population share), French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubunden.
## Output
The least popular official language of Switzerland is Romansh.

#
## Input
Question: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums.
## Output
The Dealers is an album by Mal Waldron.

#
## Input
Question: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 - December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants.
## Output
Mal Waldron was born in New York City.

#
## Input
Question: When did Hurricane Sandy hit New York City?
Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) formed October 28, 2012 and dissipated November 2, 2012. Areas affected New York, especially the New York metropolitan area.
## Output
Hurricane Sandy hit New York City on October 28, 2012.
"""


QUERY_GENERATION_CONTRIEVER_SYSTEM_PROMPT = """\
Given an original question and a series of follow-up questions and answers, generate the next logical follow-up question that targets a specific missing piece of information needed to answer the original question.
The question will be used to retrieve additional information from a knowledge base (e.g., Wikipedia).

Process:
- You receive:
  - Main question: <original question>
  - Zero or more rounds of:
    - Follow up: <previous follow up question>
    - Document: <top Wikipedia snippet>
    - Intermediate answer: <answer to the Follow-up question based on the Document>

- Your task:
  1. Ask the next logical follow-up question.
  2. Base it solely on gaps in the existing trace.
  3. Do not repeat information already covered.
  4. Make sure it targets exactly the missing fact you need to answer the original question.
  5. Output only the new question itself; no explanations or extra text.

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

# First iteration (no trace)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
## Output
Where did Andre Bloc live when he died?

# First iteration (no trace)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
## Output
Where was Karin Thomas born?

# Subsequent iteration (trace present)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did Andre Bloc live when he died?
Document: Andre Bloc (Algiers, May 23, 1896 - New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: Andre Bloc was living in New Delhi when he died.
## Output
What is the name of the major railroad related museum located in New Delhi?

# Subsequent iteration (trace present)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubunden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
## Output
What is the least popular official language of Switzerland?

# Subsequent iteration (trace present)
## Input
Main question: When did Hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums.
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 - December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
## Output
When did Hurricane Sandy hit New York City?
"""


QUERY_GENERATION_BM25_SYSTEM_PROMPT = """\
Given an original question and a series of follow-up questions and answers, generate the next logical follow-up question that targets a specific missing piece of information needed to answer the original question.
The question will be used to retrieve additional information from a knowledge base (e.g., Wikipedia).

Process:
- You receive:
  - Main question: <original question>
  - Zero or more rounds of:
    - Follow up: <previous follow up question>
    - Document: <top Wikipedia snippet>
    - Intermediate answer: <answer to the Follow-up question based on the Document>

- Your task:
  1. Ask the next logical follow-up question.
  2. Add keywords that capture the essence of the follow-up question. Also include a few synonyms of the keywords.
  3. Base it solely on gaps in the existing trace.
  4. Do not repeat information already covered.
  5. Make sure it targets exactly the missing fact you need to answer the original question.
  6. Output only the new question and the keywords/synonyms; no explanations or extra text.

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

# First iteration (no trace)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
## Output
Where did Andre Bloc live when he died? (Keywords/Synonyms: Andre Bloc, lived, residence, location, death, died)

# First iteration (no trace)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
## Output
Where was Karin Thomas born? (Keywords/Synonyms: Karin Thomas, birthplace, place of birth, born)

# Subsequent iteration (trace present)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did Andre Bloc live when he died?
Document: Andre Bloc (Algiers, May 23, 1896 - New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: Andre Bloc was living in New Delhi when he died.
## Output
What is the name of the major railroad related museum located in New Delhi? (Keywords/Synonyms: railroad, metro, train, museum, New Delhi)

# Subsequent iteration (trace present)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubunden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
## Output
What is the least popular official language of Switzerland? (Keywords/Synonyms: least, less, small, popular, spoken, prevalent, official, language, Switzerland)

# Subsequent iteration (trace present)
## Input
Main question: When did Hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums.
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 - December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
## Output
When did Hurricane Sandy hit New York City? (Keywords/Synonyms: Hurricane Sandy, New York City)
"""


QUERY_GENERATION_USER_PROMPT = (
    "Respond with a simple follow-up question that will help answer the main question, "
    "do not explain yourself or output anything else."
)


def format_intermediate_answer_prompt(question: str, document: str) -> str:
    return f"Question: {question.strip()}\nDocument: {document.strip()}"


def gen_intermediate_answer_prompt(trace: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": trace.strip(),
        },
    ]


def gen_final_answer_prompt(trace: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": ANSWER_STYLE_RULES,
        },
        {
            "role": "user",
            "content": trace.strip(),
        },
    ]


def format_query_generation_prompt(question: str, trace: str) -> str:
    body = f"Main question: {question.strip()}"
    if trace.strip():
        body += f"\n\n{trace.strip()}"
    body += f"\n\n{QUERY_GENERATION_USER_PROMPT}"
    return body


def get_query_generation_system_prompt(retriever_backend: str) -> str:
    backend = (retriever_backend or "bm25").strip().lower()
    if backend == "contriever_bge":
        return QUERY_GENERATION_CONTRIEVER_SYSTEM_PROMPT
    return QUERY_GENERATION_BM25_SYSTEM_PROMPT


def gen_contriever_query_prompt(question: str, trace: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": QUERY_GENERATION_CONTRIEVER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": format_query_generation_prompt(question, trace),
        },
    ]


def gen_bm25_query_prompt(question: str, trace: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": QUERY_GENERATION_BM25_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": format_query_generation_prompt(question, trace),
        },
    ]


def append_trace_step(trace: str, follow_up: str, document: str, intermediate_answer: str) -> str:
    parts = []
    if trace.strip():
        parts.append(trace.strip())
    parts.append(f"Follow up: {follow_up.strip()}")
    parts.append(f"Document: {document.strip()}")
    parts.append(f"Intermediate answer: {intermediate_answer.strip()}")
    return "\n".join(parts)

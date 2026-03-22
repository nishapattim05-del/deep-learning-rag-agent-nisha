# Risk Assessment

## 1. Hallucination Risk
- Risk: LLM generates answers without context
- Mitigation: Hallucination guard + strict SYSTEM_PROMPT

## 2. Poor Chunk Quality
- Risk: Bad chunks lead to poor retrieval
- Mitigation: Chunking rules (100–300 words, atomic ideas)

## 3. Duplicate Data
- Risk: Same content ingested multiple times
- Mitigation: Content-based hashing (chunk_id)

## 4. JSON Parsing Failures
- Risk: LLM returns malformed JSON
- Mitigation: Strict prompt instructions ("Respond with valid JSON only")

## 5. Retrieval Failure
- Risk: No relevant chunks found
- Mitigation: Query rewriting + fallback handling
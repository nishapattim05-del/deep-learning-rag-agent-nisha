# Integration Test Plan

## 1. Normal Query
**Input:** Explain vanishing gradient  
**Expected Output:**
- Relevant chunks retrieved
- Answer generated from corpus
- Source citations present

---

## 2. Off-topic Query (Hallucination Guard)
**Input:** History of Rome  
**Expected Output:**
- No chunks retrieved
- System returns "No relevant context found"
- No hallucinated answer

---

## 3. Duplicate Ingestion
**Input:** Upload same file twice  
**Expected Output:**
- First ingestion → chunks added
- Second ingestion → duplicates skipped
- No duplicate entries in database

---

## 4. Empty Query
**Input:** (blank input)  
**Expected Output:**
- No crash
- Graceful handling (no response or warning)

---

## 5. Cross-topic Query
**Input:** How do LSTMs improve on RNNs?  
**Expected Output:**
- Multiple chunks retrieved
- Multiple sources cited
- Answer combines multiple topics
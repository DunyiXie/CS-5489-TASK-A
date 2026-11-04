# Step 5a memory baseline

Fit: official train only, complete entity token tuples, one unique NE type. Ambiguous surfaces discarded; O is not an NE type.
Decode: exact case-sensitive words, all sentence-local candidates sorted by length descending then start ascending; skip occupied tokens. Output valid IOB2, adjacent B tags preserved. Predict never receives gold/POS/chunk.
The paper does not fully specify overlap ties/casing; these explicit project choices were frozen before any held-out score. This is not a claim of bit-for-bit historical reproduction.
src/step5a_memory.py: MemoryBaseline.fit(train_rows, label_names); predict(token_sentences); freeze_model; load_frozen; evaluate_cached. Shared Step 4 APIs and protocol are unchanged.
The full model dictionary and ambiguity counts are in config/ner-memory-baseline-v1/<contract>/model.json, with final_config.json. No large weights or new tokenizer.
Original-sentence JSONL predictions and sidecars are in results/predictions/. Scores and 2000-draw, 95% sentence bootstrap caches are in results/ner-memory-baseline-v1/<contract>/. All metrics/intervals use fractions.
Same-model reruns verify and reuse saved predictions/evaluations. If the runtime is lost AFTER a prior Step 5a run, upload your verified Step 5a ZIP using the Files panel and set RESTORE_STEP5A_ZIP in section 5a.2 to restore to a new directory. Full-corpus data are recovered from the pinned public source.
Retain every earlier ZIP and download this notebook separately. The ZIP contains the complete small baseline model but excludes raw datasets/cache.
Gate 1 pipeline checks are complete only after this notebook runs successfully; external acceptance still requires the user download and ZIP review. Next: Step 5b LR/SVM.
Reference: https://aclanthology.org/W03-0419.pdf section 3.4/Table 5. P=71.91/R=50.90/F1=59.61 are published percentages, not substituted experiment scores. No threshold forcing agreement.

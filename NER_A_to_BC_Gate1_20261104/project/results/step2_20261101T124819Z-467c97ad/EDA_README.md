# Step 2 descriptive EDA

No model training or prediction scoring.

Reference: https://aclanthology.org/W03-0419.pdf (Tables 1 and 2).

## Definitions

- entity: Strict IOB2 span, counted once at B-. End index is exclusive.
- surface: Exact tuple of original entity tokens; case-sensitive, no punctuation or whitespace normalization.
- unseen_primary: Surface absent among ALL training gold entity types. Mention rate weights occurrences; unique rate weights distinct surfaces within each type.
- unseen_typed: Surface absent from training gold entities of the SAME type; reported separately.
- casefold: Supplemental case-insensitive comparison uses str.casefold() on each token; original data are unchanged.
- capitalization: First character of the original first token is uppercase (str.isupper). Entity denominator: all mentions of that type. Sentence-O denominator: sentences whose token at index 0 has O; no punctuation skipping.
- transition: Only adjacent tokens within a sentence; no transitions across sentences. Probability divides by outgoing transitions for that label. Zero denominator is null (N/A).
- ambiguity: Exact token forms or complete entity surfaces with at least two distinct entity types WITHIN a split. O is excluded from entity types; O occurrence counts are supplemental.
- vocabulary: Includes punctuation and digits. Directed source-to-target type coverage = intersection / target vocabulary size; token coverage weights target token occurrences. Jaccard = intersection / union.
- top_entities: Top 10 per split and type, by mention count descending then token tuple ascending; plots show training entities.
- paper_sentences: Compare HF sentences directly to 14041/3250/3453; also compare HF sentences + document count to paper sentence count. DOCSTART explanation is an inference, not a reconstruction of the original files.
- test_use: Test statistics are descriptive only and must not select features, hyperparameters, preprocessing or models. No prediction scores are computed.

## Observed counts

| Split | Sentences | Tokens | Entities | PER | ORG | LOC | MISC |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 14041 | 203621 | 23499 | 6600 | 6321 | 7140 | 3438 |
| validation | 3250 | 51362 | 5942 | 1842 | 1341 | 1837 | 922 |
| test | 3453 | 46435 | 5648 | 1617 | 1661 | 1668 | 702 |

Paper count comparison passed: True.
Raw paper sentence counts include an additional document-sized difference; see paper_comparison.csv.
Use train-only observations when considering features. Test statistics are descriptive, not a selection criterion.
PNG and vector PDF figures are stored in the corresponding figures/step2_<run_id> directory.

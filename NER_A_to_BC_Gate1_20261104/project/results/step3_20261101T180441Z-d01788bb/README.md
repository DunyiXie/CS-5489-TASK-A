# Step 3 preprocessing

- 256
- True
- IOB2 labels retained from metadata; no relabeling at chunk boundaries.
- First subword of each original word only; continuation/special/padding labels are -100.
- Greedy nonoverlapping complete-word chunks based only on tokenized word lengths; never gold entities.
- Raise an error if one word exceeds the content budget or produces no subword; never silently discard it.
- Reassemble each original sentence in word order before entity metrics; do not score chunks separately.
- Whole-word chunking can cut through an entity and limits cross-chunk context; original gold labels are preserved.
- Dynamic right-padding per batch; strip alignment metadata before passing a batch to a model.
- Tokenizer preparation only: no model weights, training, predictions or F1 scoring.

The maximum length was fixed before tokenized test lengths were observed. Test transformations are integrity checks only, with no parameter selection or scoring. First-subword supervision is verified for every original word. Gold-label round trips are alignment checks, not model accuracy measurements. Chunking can reduce cross-boundary context; I-tags at a chunk start are preserved. Future metrics must reconstruct whole sentences before entity extraction. Prediction reconstruction uses word_ids and does not inspect gold labels. Arrow files and saved tokenizers stay under data/ and are excluded from the ZIP. If the runtime is lost, rerun Step 3 with the same pinned sources. Retain earlier ZIPs and download the notebook separately.

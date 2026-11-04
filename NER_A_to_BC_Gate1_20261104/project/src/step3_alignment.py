"""Lossless word/subword alignment, whole-word chunks and prediction reconstruction."""
import hashlib
import json
from collections import Counter
from numbers import Integral
from pathlib import Path

IGNORE_INDEX = -100
MAX_LENGTH = 256  # Includes special tokens; fixed before observing tokenized test lengths.
TOKENIZER_SPECS = {
    'bert_base_cased': {
        'repo_id': 'google-bert/bert-base-cased',
        'revision': 'cd5ef92a9fb2f889e972770a36d4ed042daf221e',
        'files': ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt'],
        'kwargs': {},
    },
    'roberta_base': {
        'repo_id': 'FacebookAI/roberta-base',
        'revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
        'files': ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'],
        'kwargs': {'add_prefix_space': True},
    },
}
POLICY = {
    'max_length': MAX_LENGTH,
    'includes_special_tokens': True,
    'scheme': 'IOB2 labels retained from metadata; no relabeling at chunk boundaries.',
    'supervision': 'First subword of each original word only; continuation/special/padding labels are -100.',
    'chunking': 'Greedy nonoverlapping complete-word chunks based only on tokenized word lengths; never gold entities.',
    'oversized_word': 'Raise an error if one word exceeds the content budget or produces no subword; never silently discard it.',
    'evaluation': 'Reassemble each original sentence in word order before entity metrics; do not score chunks separately.',
    'context_limit': 'Whole-word chunking can cut through an entity and limits cross-chunk context; original gold labels are preserved.',
    'padding': 'Dynamic right-padding per batch; strip alignment metadata before passing a batch to a model.',
    'model_work': 'Tokenizer preparation only: no model weights, training, predictions or F1 scoring.',
}


def load_tokenizer(key, project_root):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from step1_data import file_sha256

    spec = TOKENIZER_SPECS[key]
    snapshot = Path(snapshot_download(repo_id=spec['repo_id'], revision=spec['revision'],
        allow_patterns=spec['files'], cache_dir=str(Path(project_root) / 'data' / 'tokenizer_cache'), token=False))
    files = []
    for filename in spec['files']:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing pinned tokenizer file: {filename}')
        files.append(dict(path=filename, size_bytes=path.stat().st_size, sha256=file_sha256(path)))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True,
                                               trust_remote_code=False, **spec['kwargs'])
    tokenizer.padding_side = 'right'
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('A fast tokenizer with a padding token is required.')
    if MAX_LENGTH > tokenizer.model_max_length:
        raise ValueError('Chosen maximum exceeds the tokenizer model limit.')
    record = dict(key=key, **spec, source_files=files, tokenizer_class=type(tokenizer).__name__,
                  model_max_length=tokenizer.model_max_length, padding_side=tokenizer.padding_side,
                  special_tokens_per_sentence=tokenizer.num_special_tokens_to_add(pair=False))
    return tokenizer, record


def whole_word_ranges(piece_counts, budget):
    """Return contiguous [start,end) word ranges, with no overlap or truncation."""
    if budget < 1 or not piece_counts:
        raise ValueError('Positive content budget and at least one word are required.')
    ranges, start, used = [], 0, 0
    for index, count in enumerate(piece_counts):
        if count < 1 or count > budget:
            raise ValueError(f'Word {index} has {count} pieces, outside content budget 1..{budget}.')
        if used + count > budget:
            ranges.append((start, index))
            start, used = index, 0
        used += count
    ranges.append((start, len(piece_counts)))
    return ranges


def encode_sentence(row, sentence_index, tokenizer, label_names, max_length=MAX_LENGTH):
    words, gold = row['tokens'], row['ner_tags']
    if not words or len(words) != len(gold):
        raise ValueError('Missing words or token/gold length mismatch.')
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < len(label_names) for v in gold):
        raise ValueError('Invalid gold label index.')
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    if max_length > tokenizer.model_max_length:
        raise ValueError('Chosen max_length exceeds the model limit.')
    flat = tokenizer(words, is_split_into_words=True, add_special_tokens=False, truncation=False,
                     padding=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)
    flat_ids, word_ids = flat['input_ids'], flat.word_ids()
    if len(flat_ids) != len(word_ids) or any(w is None for w in word_ids):
        raise ValueError('Unexpected special/unmapped token in plain word encoding.')
    runs = [w for i, w in enumerate(word_ids) if i == 0 or word_ids[i-1] != w]
    if runs != list(range(len(words))):
        raise ValueError('Tokenizer omitted, reordered or fragmented a word mapping.')
    counts = Counter(word_ids)
    pieces = [counts[i] for i in range(len(words))]
    boundaries = [0]
    for count in pieces:
        boundaries.append(boundaries[-1] + count)
    ranges = whole_word_ranges(pieces, max_length - special_count)
    chunks = []
    for chunk_index, (start, end) in enumerate(ranges):
        encoded = tokenizer(words[start:end], is_split_into_words=True, add_special_tokens=True,
                            truncation=False, padding=False, return_special_tokens_mask=True)
        local_ids = encoded.word_ids()
        if [w for i,w in enumerate(local_ids) if w is not None and (i == 0 or local_ids[i-1] != w)] != list(range(end-start)):
            raise ValueError('Chunk word mapping changed.')
        content = [token for token,w in zip(encoded['input_ids'], local_ids) if w is not None]
        if content != flat_ids[boundaries[start]:boundaries[end]]:
            raise ValueError('Chunk tokenization differs from full-sentence tokenization.')
        if len(encoded['input_ids']) > max_length:
            raise ValueError('Whole-word chunk still exceeds max_length.')
        labels, mapped, previous = [], [], None
        for w, special in zip(local_ids, encoded['special_tokens_mask']):
            if (w is None) != bool(special):
                raise ValueError('Special-token mask disagrees with word mapping.')
            mapped.append(-1 if w is None else start + w)
            labels.append(int(gold[start+w]) if w is not None and w != previous else IGNORE_INDEX)
            previous = w
        chunk = {k:list(encoded[k]) for k in tokenizer.model_input_names if k in encoded}
        chunk.update(labels=labels, word_ids=mapped, sentence_index=sentence_index,
                     sentence_id=str(row['id']), chunk_index=chunk_index, word_start=start, word_end=end)
        chunks.append(chunk)
    stats = dict(sentence_index=sentence_index, sentence_id=str(row['id']), words=len(words),
                 full_length=len(flat_ids)+special_count, chunks=len(chunks),
                 multi_piece_words=sum(n > 1 for n in pieces),
                 unknown_pieces=sum(v == tokenizer.unk_token_id for v in flat_ids))
    return chunks, stats


def reassemble_word_predictions(chunks, prediction_ids, word_counts, num_labels):
    """Class ids -> original words. Ignores special/continuation/padding predictions.

    Alignment uses word_ids, not gold labels. Prediction rows must follow chunk rows.
    This also validates a gold-label round trip without computing a model metric.
    """
    if len(chunks) != len(prediction_ids):
        raise ValueError('Prediction/chunk count mismatch.')
    restored = [[None] * n for n in word_counts]
    for chunk, predictions in zip(chunks, prediction_ids):
        if len(predictions) < len(chunk['input_ids']):
            raise ValueError('Prediction row shorter than its encoded chunk.')
        sentence = chunk['sentence_index']
        if not 0 <= sentence < len(restored):
            raise ValueError('Unknown sentence index.')
        previous = -1
        selected = []
        for index, word in enumerate(chunk['word_ids']):
            if word >= 0 and word != previous:
                if not 0 <= word < len(restored[sentence]):
                    raise ValueError('Word index outside original sentence.')
                value = predictions[index]
                if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < num_labels:
                    raise ValueError('Invalid prediction at a first-subword position.')
                if restored[sentence][word] is not None:
                    raise ValueError('Duplicate prediction for an original word.')
                restored[sentence][word] = int(value)
                selected.append(word)
            previous = word
        if selected != list(range(chunk['word_start'], chunk['word_end'])):
            raise ValueError('Chunk word range differs from first-subword mapping.')
    if any(value is None for row in restored for value in row):
        raise ValueError('At least one original word has no prediction.')
    return restored


def model_features(chunk, tokenizer):
    """Keep metadata out of the model/collator input, retaining it in the Arrow sidecar."""
    return {k:chunk[k] for k in [*tokenizer.model_input_names, 'labels'] if k in chunk}


def validate_chunks(chunks, source_rows, num_labels, max_length=MAX_LENGTH):
    counts = [len(row['tokens']) for row in source_rows]
    labels = [row['ner_tags'] for row in source_rows]
    special = continuation = supervised = 0
    lengths = []
    for chunk in chunks:
        n = len(chunk['input_ids'])
        if n > max_length or n < 1 or any(len(chunk[k]) != n for k in ('attention_mask','labels','word_ids')):
            raise ValueError('Chunk lengths are inconsistent.')
        if 'token_type_ids' in chunk and len(chunk['token_type_ids']) != n:
            raise ValueError('Token-type ids have an inconsistent length.')
        if any(isinstance(w, bool) or not isinstance(w, Integral) or w < -1 for w in chunk['word_ids']):
            raise ValueError('Invalid stored word index.')
        if any(v != 1 for v in chunk['attention_mask']):
            raise ValueError('Saved chunks must be unpadded.')
        previous = -1
        for word, value in zip(chunk['word_ids'], chunk['labels']):
            first = word >= 0 and word != previous
            if not first:
                if value != IGNORE_INDEX:
                    raise ValueError('Special or continuation token carries supervision.')
                special += int(word < 0)
                continuation += int(word >= 0)
            else:
                supervised += 1
            previous = word
        lengths.append(n)
    restored = reassemble_word_predictions(chunks, [r['labels'] for r in chunks], counts, num_labels)
    if restored != labels or supervised != sum(counts):
        raise ValueError('Gold labels did not round-trip for every original word exactly once.')
    return dict(original_words=sum(counts), supervised_words=supervised, dropped_words=0,
                duplicated_words=0, masked_special_tokens=special, masked_continuation_tokens=continuation,
                max_chunk_length=max(lengths, default=0), gold_roundtrip_passed=True)


def padding_check(tokenizer, chunks):
    from transformers import DataCollatorForTokenClassification
    import numpy as np

    chosen = [min(chunks, key=lambda x:len(x['input_ids'])), max(chunks, key=lambda x:len(x['input_ids']))]
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True,
                                                  label_pad_token_id=IGNORE_INDEX, return_tensors='np')
    batch = collator([model_features(row, tokenizer) for row in chosen])
    for i, row in enumerate(chosen):
        n = len(row['input_ids'])
        if list(batch['labels'][i,:n]) != row['labels']:
            raise ValueError('Collator changed original labels.')
    masked = batch['attention_mask'] == 0
    if not np.all(batch['labels'][masked] == IGNORE_INDEX):
        raise ValueError('Padding labels must be -100.')
    return dict(batch_shape=list(batch['input_ids'].shape), padding_positions=int(masked.sum()), padding_labels_verified=True)


def records_sha256(records):
    digest = hashlib.sha256()
    for row in records:
        digest.update((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8'))
    return digest.hexdigest()


def synthetic_alignment_check(tokenizer, label_names):
    """Exercise multi-piece words, forced long-entity splitting and dynamic padding."""
    ids = {label: i for i, label in enumerate(label_names)}
    rows = [
        dict(id='synthetic-short', tokens=['Jane', 'Villanueva', 'visited', 'Hong', 'Kong', '.'],
             ner_tags=[ids[t] for t in ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O']]),
        dict(id='synthetic-long', tokens=['Jane'] + ['Villanueva']*180 + ['.'],
             ner_tags=[ids['B-PER']] + [ids['I-PER']]*180 + [ids['O']]),
        dict(id='synthetic-symbols', tokens=['U.N.', 'München', '中文', '[CLS]', '<s>', '1996-08-22'],
             ner_tags=[ids['O']]*6),
    ]
    chunks, lengths = [], []
    for index, row in enumerate(rows):
        encoded, info = encode_sentence(row, index, tokenizer, label_names, max_length=64)
        chunks.extend(encoded); lengths.append(info)
    audit = validate_chunks(chunks, rows, len(label_names), max_length=64)
    padded = padding_check(tokenizer, chunks)
    if lengths[1]['chunks'] <= 1 or audit['masked_continuation_tokens'] == 0 or padded['padding_positions'] == 0:
        raise ValueError('Synthetic test did not exercise splitting, continuation and padding.')
    restored = reassemble_word_predictions(chunks, [r['labels'] + [0]*3 for r in chunks],
                                          [len(r['tokens']) for r in rows], len(label_names))
    if restored != [r['ner_tags'] for r in rows]:
        raise ValueError('Synthetic reconstruction with padded predictions failed.')
    return dict(synthetic_only=True, forced_max_length=64, sentences=lengths, audit=audit, padding=padded)


def prepare_tokenizer_dataset(dataset, tokenizer, label_names, destination, key):
    from datasets import Dataset, DatasetDict, load_from_disk
    from step1_data import file_sha256

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Output exists. Reinitialize Step 3 to create a fresh run_id.')
    prepared, summaries, histograms, overflow, examples = {}, [], [], [], []
    expected_hashes = {}
    for split, source in dataset.items():
        records, lengths = [], Counter()
        multi = unknown = long_count = 0
        for index, row in enumerate(source):
            chunks, info = encode_sentence(row, index, tokenizer, label_names)
            records.extend(chunks)
            lengths[info['full_length']] += 1
            multi += info['multi_piece_words']; unknown += info['unknown_pieces']
            if info['full_length'] > MAX_LENGTH:
                long_count += 1
                overflow.append(dict(tokenizer=key, split=split, **info))
            if split == 'train' and (index < 2 or (info['chunks'] > 1 and not any(e['example_kind']=='long_train' for e in examples))):
                for chunk in chunks:
                    for pos, (piece, word, label) in enumerate(zip(tokenizer.convert_ids_to_tokens(chunk['input_ids']),chunk['word_ids'],chunk['labels'])):
                        examples.append(dict(tokenizer=key, example_kind='first_train' if index<2 else 'long_train',
                            sentence_index=index, chunk_index=chunk['chunk_index'], position=pos, subword=piece,
                            word_index=word, original_word='' if word<0 else row['tokens'][word],
                            label_id=label, label='IGNORED' if label==IGNORE_INDEX else label_names[label]))
            if (index+1) % 5000 == 0:
                print(f'{key}/{split}: aligned {index+1}/{len(source)} sentences')
        validation = validate_chunks(records, source, len(label_names))
        padded = padding_check(tokenizer, records)
        expected_hashes[split] = records_sha256(records)
        prepared[split] = Dataset.from_list(records)
        summaries.append(dict(tokenizer=key,split=split,sentences=len(source),chunks=len(records),
            over_max_length_sentences=long_count,max_full_length=max(lengths),
            multi_piece_words=multi,unknown_pieces=unknown,**validation,**padded,
            ordered_alignment_sha256=expected_hashes[split]))
        histograms.extend(dict(tokenizer=key,split=split,full_length=n,sentences=count) for n,count in sorted(lengths.items()))
    result = DatasetDict(prepared)
    result.save_to_disk(str(destination / 'dataset'))
    tokenizer.save_pretrained(str(destination / 'tokenizer'))
    reloaded = load_from_disk(str(destination / 'dataset'))
    for split, source in dataset.items():
        if records_sha256(reloaded[split]) != expected_hashes[split]:
            raise ValueError('Alignment files changed after save/read-back.')
        validate_chunks(reloaded[split], source, len(label_names))
    files = [dict(path=p.relative_to(destination).as_posix(),size_bytes=p.stat().st_size,sha256=file_sha256(p))
             for p in sorted(destination.rglob('*')) if p.is_file()]
    return dict(key=key, path=str(destination), summaries=summaries, length_histograms=histograms,
                overflow=overflow, alignment_examples=examples, saved_files=files, read_back_verified=True)

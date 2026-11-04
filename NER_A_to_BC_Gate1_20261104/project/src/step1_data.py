"""Pinned CoNLL-2003 loading and validation. Run real data work in Colab only."""
import hashlib
import json
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

REPO_ID = 'tomaarsen/conll2003'
REVISION = '3a0c0677d3177b6122fceed3239dda7a5de6a514'
EXPECTED_ROWS = {'train': 14041, 'validation': 3250, 'test': 3453}
TAG_FIELDS = ('pos_tags', 'chunk_tags', 'ner_tags')
REQUIRED_FIELDS = ('id', 'tokens', *TAG_FIELDS)
EXPECTED_NER_LABELS = {'O'} | {f'{p}-{t}' for p in ('B', 'I') for t in ('PER', 'ORG', 'LOC', 'MISC')}
PARQUET_FILES = {s: f'data/{s}-00000-of-00001.parquet' for s in EXPECTED_ROWS}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_pinned_dataset(project_root):
    """Download public data, without auth or remote loading scripts; retain file hashes."""
    import datasets
    from huggingface_hub import snapshot_download

    if datasets.__version__ != '4.0.0':
        raise RuntimeError('Use datasets==4.0.0; install it and restart the Python session.')
    cache = Path(project_root) / 'data' / 'hf_cache'
    snapshot = Path(snapshot_download(
        repo_id=REPO_ID, repo_type='dataset', revision=REVISION,
        allow_patterns=[*PARQUET_FILES.values(), 'README.md'],
        cache_dir=str(cache / 'hub'), token=False,
    ))
    files = {split: str(snapshot / name) for split, name in PARQUET_FILES.items()}
    source_files = []
    for name in [*PARQUET_FILES.values(), 'README.md']:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f'Pinned source file missing: {name}')
        source_files.append({'path': name, 'size_bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    dataset = datasets.load_dataset('parquet', data_files=files,
                                    cache_dir=str(cache / 'arrow'), token=False)
    provenance = {
        'requested_source': 'eriktks/conll2003',
        'actual_source': REPO_ID, 'revision': REVISION, 'format': 'parquet',
        'source_url': f'https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}',
        'reason': 'Development document permits Parquet fallback; datasets 4.0 removes loading scripts.',
        'license_as_stated_by_card': 'other',
        'source_files': source_files,
        'preprocessing': 'None: no shuffle, split, filtering, label conversion or token normalization.',
        'parity_scope': 'Split sizes, metadata, all-row structure/IOB2 and ordered content hashes checked. '
                        'Byte-for-byte equivalence to eriktks/conll2003 has not been established.',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    return dataset, provenance


def label_names(split):
    names = {}
    for field in TAG_FIELDS:
        feature = split.features[field]
        values = getattr(getattr(feature, 'feature', None), 'names', None)
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{field} must have unique ClassLabel names in feature metadata.')
        names[field] = list(values)
    if set(names['ner_tags']) != EXPECTED_NER_LABELS:
        raise ValueError('NER metadata must contain O and B/I labels for PER, ORG, LOC, MISC.')
    return names


def audit_dataset(dataset, expected_rows=None):
    """Reject corrupt/IOB1 data instead of silently repairing labels or dropping rows.

    expected_rows is injectable for tiny offline fixtures; real runs use EXPECTED_ROWS.
    Hash input: UTF-8 compact JSON for each core record, decoded tag strings, plus LF.
    """
    expected_rows = EXPECTED_ROWS if expected_rows is None else expected_rows
    if set(dataset) != set(expected_rows):
        raise ValueError('Unexpected dataset splits.')
    report = {'splits': {}, 'label_names': None, 'iob2_violations': 0,
              'hash_format': 'core-fields-decoded-tags-jsonl-utf8-v1'}
    for split_name, expected in expected_rows.items():
        split = dataset[split_name]
        if len(split) != expected:
            raise ValueError(f'{split_name}: expected {expected} rows, received {len(split)}.')
        if not set(REQUIRED_FIELDS).issubset(split.column_names):
            raise ValueError(f'{split_name}: missing required columns.')
        names = label_names(split)
        if report['label_names'] is None:
            report['label_names'] = names
        elif names != report['label_names']:
            raise ValueError(f'{split_name}: label metadata differs between splits.')
        has_documents = {'document_id', 'sentence_id'}.issubset(split.column_names)
        digest = hashlib.sha256()
        tokens_total = 0
        previous_doc, previous_sentence = None, None
        document_count = 0
        for index, row in enumerate(split):
            where = f'{split_name}[{index}]'
            if row['id'] != str(index):
                raise ValueError(f'{where}: id/order changed (expected id={index}).')
            tokens = row['tokens']
            if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
                raise ValueError(f'{where}: empty or invalid token list.')
            if '-DOCSTART-' in tokens:
                raise ValueError(f'{where}: unexpected DOCSTART sentinel.')
            decoded = {}
            for field in TAG_FIELDS:
                tags = row[field]
                if not isinstance(tags, list) or len(tags) != len(tokens):
                    raise ValueError(f'{where}: token/{field} length mismatch.')
                if any(isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < len(names[field]) for t in tags):
                    raise ValueError(f'{where}: invalid {field} index.')
                decoded[field] = [names[field][int(t)] for t in tags]
            previous = 'O'
            for position, tag in enumerate(decoded['ner_tags']):
                if tag.startswith('I-') and previous not in ('B-' + tag[2:], tag):
                    raise ValueError(f'{where} token {position}: illegal IOB2 transition {previous} -> {tag}.')
                previous = tag
            if has_documents:
                doc, sentence = row['document_id'], row['sentence_id']
                if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (doc, sentence)):
                    raise ValueError(f'{where}: invalid document/sentence id.')
                if previous_doc is None or doc != previous_doc:
                    if sentence != 0 or (previous_doc is not None and doc != previous_doc + 1):
                        raise ValueError(f'{where}: document order is not contiguous.')
                    document_count += 1
                elif sentence != previous_sentence + 1:
                    raise ValueError(f'{where}: sentence order is not contiguous.')
                previous_doc, previous_sentence = doc, sentence
            record = {'id': row['id'], 'tokens': tokens, **decoded}
            digest.update((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
            tokens_total += len(tokens)
        report['splits'][split_name] = {
            'rows': len(split), 'tokens': tokens_total, 'ordered_core_sha256': digest.hexdigest(),
            'id_order_verified': True, 'document_order_verified': has_documents,
            'documents': document_count if has_documents else None,
        }
    return report


def save_and_verify(dataset, project_root, run_id, report):
    """Keep Arrow data on the runtime; verify persistence without putting data in ZIPs."""
    from datasets import load_from_disk

    destination = Path(project_root) / 'data' / f'conll2003_{run_id}'
    if destination.exists():
        raise FileExistsError('This run was already saved. Rerun initialization for a new run_id.')
    dataset.save_to_disk(str(destination))
    restored = load_from_disk(str(destination))
    check = audit_dataset(restored)
    if check != report:
        raise ValueError('Saved dataset differs after read-back.')
    # Includes auxiliary columns and feature metadata, in addition to ordered core hashes.
    files = [{'path': p.relative_to(destination).as_posix(), 'size_bytes': p.stat().st_size,
              'sha256': file_sha256(p)} for p in sorted(destination.rglob('*')) if p.is_file()]
    return {'path': str(destination), 'read_back_verified': True, 'files': files}

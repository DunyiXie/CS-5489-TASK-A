"""Train-only phrase memory baseline; deterministic decoding and reusable evaluation."""
import hashlib
import json
from collections import Counter, defaultdict
from numbers import Integral
from pathlib import Path

import metrics
import prediction_io
from experiment_protocol import canonical_bytes, fingerprint

VERSION = 'ner-memory-baseline-v1'
POLICY = {
    'version': VERSION,
    'fit': 'Official train only; memorize complete gold entity token tuples, never subphrases.',
    'unique': 'Keep a surface only when its complete training entities have exactly one NE type; discard all multi-type surfaces regardless of frequency. O is not an entity type.',
    'matching': 'Exact original tokens, case-sensitive; no normalization, substring matches or external gazetteers.',
    'overlap': 'Collect all sentence-local matches; accept by descending token length, then ascending start offset, skipping any occupied token. Finally sort by start.',
    'tie_break': 'Equal-length overlapping matches prefer the earlier start. A retained surface has only one type.',
    'output': 'IOB2: B at each accepted entity start, I inside, O elsewhere. Adjacent entities remain separate. Never cross sentence boundaries.',
    'selection': 'One preregistered configuration; no hyperparameters, CV, frequency cutoff or validation/test-based changes.',
    'interpretation': 'Implements the paper description. Case sensitivity and equal-length/crossing-overlap resolution are explicit project choices; exact identity to the historical executable is not established.',
    'reference': 'https://aclanthology.org/W03-0419.pdf (section 3.4, Table 5)',
}
PAPER = {'precision': 0.7191, 'recall': 0.5090, 'f1': 0.5961}
BOOTSTRAP = {'n_resamples': 2000, 'confidence': 0.95, 'seed': 2026}


def _tokens(tokens):
    if not isinstance(tokens, (list, tuple)) or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError('Expected original nonempty token strings, not raw text or labelled rows.')
    return list(tokens)


def _labels(label_names):
    if len(label_names) != len(metrics.LABELS) or set(label_names) != metrics.LABELS:
        raise ValueError('Use the complete metadata-derived label order.')


class MemoryBaseline:
    """Predict takes tokens only. No gold, POS, chunk labels or split context."""

    def __init__(self, payload):
        if payload.get('version') != VERSION or payload.get('policy') != POLICY:
            raise ValueError('Memory model version/policy mismatch.')
        _labels(payload['label_names'])
        self.payload = json.loads(json.dumps(payload))
        self._trie = {}
        seen = set()
        for item in payload['entries']:
            tokens = tuple(_tokens(item['tokens']))
            if not tokens or tokens in seen or item['type'] not in metrics.TYPES or item['mentions'] < 1:
                raise ValueError('Invalid or duplicate dictionary entry.')
            seen.add(tokens)
            node = self._trie
            for token in tokens:
                node = node.setdefault(token, {})
            node[None] = item['type']

    @classmethod
    def fit(cls, train_rows, label_names):
        _labels(label_names)
        counts = defaultdict(Counter)
        digest = hashlib.sha256()
        n_sentences = n_words = 0
        ids = set()
        for row in train_rows:
            tokens = _tokens(row['tokens'])
            identity = str(row['id'])
            if not tokens or identity in ids or not identity or len(tokens) != len(row['ner_tags']):
                raise ValueError('Invalid training row/id/length.')
            ids.add(identity)
            if any(not isinstance(v, Integral) or isinstance(v, bool) or not 0 <= v < len(label_names) for v in row['ner_tags']):
                raise ValueError('Invalid metadata label index.')
            gold = [label_names[v] for v in row['ner_tags']]
            for start, end, kind in metrics.entity_spans(gold):
                counts[tuple(tokens[start:end])][kind] += 1
            digest.update(canonical_bytes({'id': identity, 'tokens': tokens, 'gold': gold}))
            n_sentences += 1
            n_words += len(tokens)
        if not n_sentences:
            raise ValueError('Training rows must not be empty.')
        entries, ambiguous = [], []
        for tokens, types in sorted(counts.items()):
            if len(types) == 1:
                kind = next(iter(types))
                entries.append(dict(tokens=list(tokens), type=kind, mentions=types[kind]))
            else:
                ambiguous.append(dict(tokens=list(tokens), type_counts=dict(sorted(types.items()))))
        summary = dict(train_sentences=n_sentences, train_words=n_words,
            train_entity_mentions=sum(sum(c.values()) for c in counts.values()),
            distinct_entity_surfaces=len(counts), retained_surfaces=len(entries),
            ambiguous_surfaces=len(ambiguous), retained_mentions=sum(e['mentions'] for e in entries),
            excluded_ambiguous_mentions=sum(sum(e['type_counts'].values()) for e in ambiguous),
            max_retained_length=max((len(e['tokens']) for e in entries), default=0))
        return cls(dict(version=VERSION, policy=POLICY, label_names=list(label_names),
            train_sha256=digest.hexdigest(), summary=summary, entries=entries, ambiguous=ambiguous))

    def predict_sentence(self, tokens):
        tokens = _tokens(tokens)
        candidates = []
        for start in range(len(tokens)):
            node = self._trie
            for end in range(start, len(tokens)):
                node = node.get(tokens[end])
                if node is None:
                    break
                if None in node:
                    candidates.append((start, end + 1, node[None]))
        occupied = [False] * len(tokens)
        accepted = []
        for start, end, kind in sorted(candidates, key=lambda m: (-(m[1] - m[0]), m[0])):
            if not any(occupied[start:end]):
                occupied[start:end] = [True] * (end - start)
                accepted.append((start, end, kind))
        tags = ['O'] * len(tokens)
        for start, end, kind in sorted(accepted):
            tags[start] = 'B-' + kind
            tags[start + 1:end] = ['I-' + kind] * (end - start - 1)
        return tags

    def predict(self, token_sentences):
        return [self.predict_sentence(tokens) for tokens in token_sentences]


def _immutable_json(path, value):
    path = Path(path)
    content = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing frozen/cached artifact differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(content)
    if path.read_bytes() != content:
        raise ValueError('Artifact read-back failed.')


def freeze_model(project_root, model, protocol_sha256, source_hashes, data_source):
    """Persist the dictionary and settings before even validation predictions."""
    contract = dict(version=VERSION, policy=POLICY, model_sha256=fingerprint(model.payload),
        protocol_sha256=protocol_sha256, source_code_sha256s=source_hashes,
        # Observation time changes on pinned-source recovery, not model identity.
        data_source={k:v for k,v in data_source.items() if k != 'recorded_at_utc'}, bootstrap=BOOTSTRAP,
        final_comparison_list=['memory baseline validation', 'memory baseline test', 'historical paper baseline (context only)'],
        train_only=True, tuning=False, frozen_before_validation_and_test=True)
    identity = fingerprint(contract)
    directory = Path(project_root)/'config'/VERSION/identity
    _immutable_json(directory/'model.json', model.payload)
    _immutable_json(directory/'final_config.json', contract)
    receipt = dict(path=str(directory), contract_sha256=identity, model_sha256=contract['model_sha256'],
                   protocol_sha256=protocol_sha256, read_back_verified=True)
    load_frozen(receipt)
    return receipt


def load_frozen(receipt):
    path = Path(receipt['path'])
    contract_bytes = (path/'final_config.json').read_bytes()
    model_bytes = (path/'model.json').read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != receipt['contract_sha256'] or hashlib.sha256(model_bytes).hexdigest() != receipt['model_sha256']:
        raise ValueError('Frozen configuration or model checksum changed.')
    contract = json.loads(contract_bytes)
    if contract['model_sha256'] != receipt['model_sha256'] or contract['protocol_sha256'] != receipt['protocol_sha256'] or contract['policy'] != POLICY or contract['bootstrap'] != BOOTSTRAP:
        raise ValueError('Frozen model contract mismatch.')
    # A restored cache cannot silently run under edited implementation code.
    source_root = Path(__file__).parent
    for name, digest in contract['source_code_sha256s'].items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source differs: ' + name)
    return MemoryBaseline(json.loads(model_bytes)), contract


def evaluate_cached(project_root, receipt, split, rows, label_names):
    """Reuse verified saved predictions/scores on reruns; never refit on dev/test."""
    if split not in ('validation', 'test'):
        raise ValueError('Only held-out validation or final test scoring is supported.')
    model, contract = load_frozen(receipt)
    if list(label_names) != model.payload['label_names']:
        raise ValueError('Label metadata changed after freezing.')
    root = Path(project_root)
    system = 'memory_' + receipt['contract_sha256'][:20]
    path = root/'results'/'predictions'/(system + '_' + split + '.jsonl')
    cache = root/'results'/VERSION/receipt['contract_sha256']/(split + '_evaluation.json')
    reused_predictions = path.exists()
    reused_evaluation = cache.exists()
    if reused_evaluation and not reused_predictions:
        raise ValueError('Saved score exists without its predictions; restore the complete ZIP.')
    if not reused_predictions:
        # The only input reaching predict is the original word list.
        raw = model.predict([row['tokens'] for row in rows])
        prediction_io.write_predictions(path.parent, system, split, rows, raw, label_names, receipt['protocol_sha256'])
    records, meta = prediction_io.load_predictions(path, rows, label_names)
    if meta['system'] != system or meta['split'] != split or meta['protocol_sha256'] != receipt['protocol_sha256'] or meta['prediction_iob2_repairs'] != 0:
        raise ValueError('Unexpected memory-baseline prediction metadata.')
    identity = dict(contract_sha256=receipt['contract_sha256'], model_sha256=receipt['model_sha256'],
        protocol_sha256=receipt['protocol_sha256'], split=split, predictions_sha256=meta['sha256'],
        bootstrap=BOOTSTRAP)
    if reused_evaluation:
        wrapper = json.loads(cache.read_text(encoding='utf-8'))
        result = wrapper['result']
        if wrapper['sha256'] != fingerprint(result) or result['identity'] != identity:
            raise ValueError('Saved evaluation identity/checksum mismatch.')
    else:
        ci = metrics.bootstrap_ci([r['gold_tags'] for r in records], [r['raw_pred_tags'] for r in records], **BOOTSTRAP)
        result = dict(identity=identity, metrics=ci['point'], bootstrap=ci)
        _immutable_json(cache, dict(result=result, sha256=fingerprint(result)))
    return result, dict(prediction_path=str(path), metadata_path=str(path.with_suffix('.meta.json')),
        evaluation_path=str(cache), prediction_sha256=meta['sha256'],
        evaluation_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        reused_predictions=reused_predictions, reused_evaluation=reused_evaluation,
        sentences=meta['sentences'], words=meta['words'], read_back_verified=True)


def dictionary_tables(model):
    summary = model.payload['summary']
    type_table = []
    lengths = Counter()
    for kind in metrics.TYPES:
        entries = [e for e in model.payload['entries'] if e['type'] == kind]
        type_table.append(dict(type=kind, retained_surfaces=len(entries),
            retained_mentions=sum(e['mentions'] for e in entries),
            excluded_ambiguous_mentions=sum(e['type_counts'].get(kind, 0) for e in model.payload['ambiguous'])))
        for e in entries:
            lengths[(kind, len(e['tokens']))] += 1
    if sum(r['retained_mentions'] + r['excluded_ambiguous_mentions'] for r in type_table) != summary['train_entity_mentions']:
        raise ValueError('Dictionary accounting mismatch.')
    return type_table, [dict(type=k, token_length=n, retained_surfaces=v) for (k,n),v in sorted(lengths.items())]


def toy_checks():
    labels = ['I-PER','B-LOC','O','I-ORG','B-PER','I-LOC','B-ORG','B-MISC','I-MISC']
    examples = [(['New','York'],['B-LOC','I-LOC']), (['York'],['B-ORG']),
        (['York','University','Press'],['B-ORG','I-ORG','I-ORG']),
        (['Jordan'],['B-PER']), (['Jordan'],['B-LOC']),
        (['Ana'],['B-PER']), (['Ana'],['O']),
        (['A','B'],['B-LOC','I-LOC']), (['B','C'],['B-ORG','I-ORG'])]
    rows = [dict(id=str(i),tokens=t,ner_tags=[labels.index(x) for x in y]) for i,(t,y) in enumerate(examples)]
    model = MemoryBaseline.fit(rows, labels)
    cases = [
        ('longer_at_same_start',['New','York'],['B-LOC','I-LOC']),
        ('longer_crossing_later_start',['New','York','University','Press'],['O','B-ORG','I-ORG','I-ORG']),
        ('ambiguous_discarded',['Jordan'],['O']),
        ('case_sensitive',['new','York'],['O','B-ORG']),
        ('adjacent_entities',['Ana','Ana'],['B-PER','B-PER']),
        ('no_substring',['Anabel'],['O']),
        ('no_invented_subphrase',['University'],['O']),
        ('equal_length_earlier_start',['A','B','C'],['B-LOC','I-LOC','O']),
        ('unseen_all_o',['Nobody','here'],['O','O']),
        ('empty_input',[],[]),
    ]
    checked = []
    for name,tokens,expected in cases:
        actual = model.predict_sentence(tokens)
        if actual != expected:
            raise ValueError('Synthetic decoder failed: ' + name)
        metrics.entity_spans(actual)
        checked.append(dict(case=name, tokens=tokens, expected=expected, actual=actual, passed=True))
    if model.predict([['New'],['York']]) != [['O'],['B-ORG']]:
        raise ValueError('Decoder crossed a sentence boundary.')
    return dict(synthetic_only=True, cases=checked, sentence_boundary_passed=True,
                permuted_metadata_passed=True, o_is_not_entity_type_passed=True)

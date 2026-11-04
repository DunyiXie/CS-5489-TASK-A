"""Shared prediction JSONL contract, integrity checks and paired-file comparison."""
import hashlib
import json
import re
from pathlib import Path
from metrics import LABELS, METRIC_CONTRACT, canonicalize_predictions, entity_spans, paired_bootstrap

SCHEMA = {
    'version':'ner-word-predictions-v1',
    'filename':'<system>_<split>.jsonl plus .meta.json sidecar',
    'fields':{'sentence_id':'unique string; original dataset id', 'tokens':'original word strings in order',
              'gold_tags':'metadata-decoded, valid IOB2 tags', 'pred_tags':'canonical IOB2 predictions',
              'raw_pred_tags':'unaltered model tag strings before the fixed orphan-I rule'},
    'alignment':'Exactly one row per original sentence, one tag per original word. Reassemble Transformer chunks first.',
    'sidecar':'schema/metric version, system, split, protocol hash, counts and JSONL SHA-256; no overwrite.',
    'test_use':'Write final test predictions only after final settings are frozen; re-use saved files for bootstrap, never retrain during resampling.',
}


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def validate_records(records, expected_rows=None, label_names=None):
    if not isinstance(records,list) or not records:
        raise ValueError('Prediction records must be a nonempty list.')
    seen=set()
    for row in records:
        if not isinstance(row,dict) or set(row)!=set(SCHEMA['fields']):
            raise ValueError('Prediction row fields do not match the versioned schema.')
        identity=row['sentence_id']; tokens=row['tokens']
        if not isinstance(identity,str) or not identity or identity in seen:
            raise ValueError('Missing, non-string or duplicate sentence id.')
        seen.add(identity)
        if not isinstance(tokens,list) or not tokens or any(not isinstance(t,str) or not t for t in tokens):
            raise ValueError('Invalid original token list.')
        for field in ('gold_tags','pred_tags','raw_pred_tags'):
            if not isinstance(row[field],list) or len(row[field])!=len(tokens):
                raise ValueError('Prediction/gold/token length mismatch.')
        entity_spans(row['gold_tags']); entity_spans(row['pred_tags'])
        canonical,_=canonicalize_predictions([row['raw_pred_tags']])
        if canonical[0]!=row['pred_tags']:
            raise ValueError('Saved prediction differs from canonical raw prediction.')
    if expected_rows is not None:
        if label_names is None or set(label_names)!=LABELS or len(label_names)!=len(LABELS):
            raise ValueError('Expected-data validation requires the metadata label order.')
        if len(expected_rows)!=len(records):
            raise ValueError('Prediction file misses or adds source sentences.')
        for actual,source in zip(records,expected_rows):
            if actual['sentence_id']!=str(source['id']) or actual['tokens']!=source['tokens'] or actual['gold_tags']!=[label_names[v] for v in source['ner_tags']]:
                raise ValueError('Prediction file ids/tokens/gold/order differ from source data.')
    return dict(sentences=len(records),words=sum(len(r['tokens']) for r in records))


def write_predictions(directory,system,split,source_rows,raw_predictions,label_names,protocol_sha256):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',system) or split not in ('train','validation','test','synthetic'):
        raise ValueError('Use a simple system name and known split.')
    if not re.fullmatch(r'[0-9a-f]{64}',protocol_sha256):
        raise ValueError('Provide the frozen protocol SHA-256.')
    if len(source_rows)!=len(raw_predictions):
        raise ValueError('Prediction/source sentence counts differ.')
    canonical,repairs=canonicalize_predictions(raw_predictions)
    records=[dict(sentence_id=str(row['id']),tokens=list(row['tokens']),
        gold_tags=[label_names[v] for v in row['ner_tags']],pred_tags=pred,raw_pred_tags=list(raw))
        for row,raw,pred in zip(source_rows,raw_predictions,canonical)]
    counts=validate_records(records,source_rows,label_names)
    path=Path(directory)/(system+'_'+split+'.jsonl')
    sidecar=path.with_suffix('.meta.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Do not overwrite predictions; use a unique system/run name.')
    content=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records).encode('utf-8')
    metadata=dict(schema=SCHEMA['version'],metric_contract=METRIC_CONTRACT['version'],system=system,split=split,
                  protocol_sha256=protocol_sha256,sha256=_digest(content),prediction_iob2_repairs=repairs,**counts)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: stream.write(content)
    with sidecar.open('x',encoding='utf-8') as stream: json.dump(metadata,stream,ensure_ascii=False,indent=2)
    restored,restored_meta=load_predictions(path,source_rows,label_names)
    if restored!=records or restored_meta!=metadata:
        raise ValueError('Prediction JSONL save/read-back mismatch.')
    return dict(path=str(path),metadata_path=str(sidecar),**metadata,read_back_verified=True)


def load_predictions(path,expected_rows=None,label_names=None):
    path=Path(path)
    metadata=json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    content=path.read_bytes()
    if metadata.get('schema')!=SCHEMA['version'] or metadata.get('metric_contract')!=METRIC_CONTRACT['version'] or metadata.get('sha256')!=_digest(content):
        raise ValueError('Prediction version or file checksum mismatch.')
    records=[json.loads(line) for line in content.decode('utf-8').splitlines()]
    counts=validate_records(records,expected_rows,label_names)
    repairs=canonicalize_predictions([r['raw_pred_tags'] for r in records])[1]
    if any(metadata.get(k)!=v for k,v in counts.items()) or metadata.get('prediction_iob2_repairs')!=repairs:
        raise ValueError('Prediction sidecar counts mismatch.')
    if not re.fullmatch(r'[0-9a-f]{64}',metadata.get('protocol_sha256','')):
        raise ValueError('Missing protocol fingerprint.')
    return records,metadata


def compare_prediction_files(path_a,path_b,**bootstrap_options):
    a,ma=load_predictions(path_a); b,mb=load_predictions(path_b)
    if ma['split']!=mb['split'] or ma['protocol_sha256']!=mb['protocol_sha256'] or len(a)!=len(b):
        raise ValueError('Paired files must have the same split, protocol and sentence count.')
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in ('sentence_id','tokens','gold_tags')):
            raise ValueError('Paired files have different ids/tokens/gold/order.')
    result=paired_bootstrap([r['gold_tags'] for r in a],[r['raw_pred_tags'] for r in a],
                            [r['raw_pred_tags'] for r in b],**bootstrap_options)
    result['system_a']=ma['system']; result['system_b']=mb['system']
    result['prediction_file_sha256']={'a':ma['sha256'],'b':mb['sha256']}
    return result

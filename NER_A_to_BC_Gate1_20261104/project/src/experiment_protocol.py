"""Contiguous document-safe CV folds and an immutable pre-training contract."""
import hashlib
import json
from numbers import Integral
from pathlib import Path
from metrics import METRIC_CONTRACT
from prediction_io import SCHEMA

PROTOCOL_VERSION='ner-experiment-v1'
SEEDS=[42,52,62]
BOOTSTRAP_SEED=2026


def canonical_bytes(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _document_starts(rows):
    starts=[]; seen=set(); previous=None
    for i,row in enumerate(rows):
        doc=row.get('document_id')
        if isinstance(doc,bool) or not isinstance(doc,Integral):
            raise ValueError('Document-safe folds require integer document_id metadata.')
        if i==0 or doc!=previous:
            if doc in seen:
                raise ValueError('A document appears in noncontiguous sentence runs.')
            seen.add(doc); starts.append(i)
        previous=doc
    return starts


def make_contiguous_folds(train_rows,n_splits=5):
    """Balance sentence counts using nearest valid document boundary, without shuffle.

    Only ids/document ids are read. No tags, words, dev or test data select cuts.
    Tie: earlier boundary. Each remaining fold must retain >= 1 whole document.
    """
    if isinstance(n_splits,bool) or not isinstance(n_splits,Integral) or n_splits<2:
        raise ValueError('n_splits must be an integer >= 2.')
    starts=_document_starts(train_rows); n=len(train_rows)
    if len(starts)<n_splits:
        raise ValueError('Fewer documents than requested folds.')
    cut_docs=[0]
    for i in range(1,n_splits):
        choices=range(cut_docs[-1]+1,len(starts)-(n_splits-i)+1)
        cut_docs.append(min(choices,key=lambda j:(abs(starts[j]*n_splits-n*i),starts[j])))
    bounds=[starts[j] for j in cut_docs]+[n]
    folds=[]
    for fold,(start,end) in enumerate(zip(bounds,bounds[1:])):
        valid=list(range(start,end)); train=list(range(start))+list(range(end,n))
        train_docs={int(train_rows[i]['document_id']) for i in train}
        valid_docs={int(train_rows[i]['document_id']) for i in valid}
        folds.append(dict(fold=fold,train_indices=train,validation_indices=valid,
            validation_start=start,validation_end_exclusive=end,train_sentences=len(train),validation_sentences=len(valid),
            train_documents=len(train_docs),validation_documents=len(valid_docs),shared_documents=len(train_docs & valid_docs)))
    plan=dict(split='train',n_splits=int(n_splits),shuffle=False,sentence_count=n,document_count=len(starts),
        algorithm='Nearest equal-sentence target at a document boundary; earlier tie; preserve >=1 document per remaining fold.',
        rows=[dict(index=i,sentence_id=str(r['id']),document_id=int(r['document_id'])) for i,r in enumerate(train_rows)],folds=folds)
    verify_fold_plan(plan,train_rows)
    return plan


def verify_fold_plan(plan,train_rows):
    n=len(train_rows)
    if plan['split']!='train' or plan['sentence_count']!=n or plan['shuffle'] is not False:
        raise ValueError('Fold plan has the wrong source split/count/shuffle rule.')
    expected=[dict(index=i,sentence_id=str(r['id']),document_id=int(r['document_id'])) for i,r in enumerate(train_rows)]
    if plan['rows']!=expected or len({r['sentence_id'] for r in expected})!=n:
        raise ValueError('Fold plan ids or order differ from training data.')
    if plan['document_count']!=len(_document_starts(train_rows)) or len(plan['folds'])!=plan['n_splits']:
        raise ValueError('Document/fold count mismatch.')
    all_indices=set(range(n)); held_out=[]; last_end=0
    for k,fold in enumerate(plan['folds']):
        start,end=fold['validation_start'],fold['validation_end_exclusive']
        if fold['fold']!=k or start!=last_end or not start<end<=n:
            raise ValueError('Validation blocks must be consecutive, disjoint and nonempty.')
        valid=fold['validation_indices']; train=fold['train_indices']
        if valid!=list(range(start,end)) or train!=sorted(all_indices-set(valid)):
            raise ValueError('Fold indices are not an exact train/validation partition.')
        td={train_rows[i]['document_id'] for i in train}; vd={train_rows[i]['document_id'] for i in valid}
        if td & vd or fold['shared_documents']!=0:
            raise ValueError('A document leaks across the fold boundary.')
        if (len(train),len(valid),len(td),len(vd))!=(fold['train_sentences'],fold['validation_sentences'],fold['train_documents'],fold['validation_documents']):
            raise ValueError('Fold summary counts disagree with indices.')
        held_out.extend(valid); last_end=end
    if held_out!=list(range(n)):
        raise ValueError('Each training sentence must be held out exactly once.')
    return dict(valid=True,held_out_once=n,shared_documents=0,n_splits=plan['n_splits'])


def build_protocol(accepted_data,label_mapping,fold_plans,source_hashes):
    """No runtime paths/timestamps: identical inputs produce the same contract hash."""
    labels=label_mapping['label_names']
    return {
        'version':PROTOCOL_VERSION,'status':'frozen experimental design; model-specific grids must be registered before their search',
        'dataset':{'source':accepted_data['source']['actual_source'],'revision':accepted_data['source']['revision'],
                   'source_files':accepted_data['source']['source_files'],'audit':accepted_data['audit']},
        'label_names':labels,'metric_contract':METRIC_CONTRACT,'prediction_schema':SCHEMA,
        'fold_plan_sha256':{name:fingerprint(plan) for name,plan in fold_plans.items()},
        'public_source_sha256':source_hashes,
        'seeds':{'classical':42,'random_search':42,'transformer':SEEDS,'bootstrap':BOOTSTRAP_SEED,'representation_sampling':42},
        'split_usage':{
            'train':'All fitting. Classical hyperparameters use the saved 5 contiguous document-safe training folds.',
            'validation':'Compare methods/features; choose Transformer hyperparameters/checkpoints/early stopping; perform all error analysis.',
            'test':'Only after final settings/checkpoints and comparison list are frozen. Score each final checkpoint once; save predictions for all later uncertainty analysis. No tuning or error-driven changes.'},
        'classical':{'primary_folds':5,'selection':'Mean of held-out fold micro entity F1, not token accuracy; report fold std (ddof=1).',
            'tie_break':'First configuration in the pre-registered simpler-to-more-complex search order.',
            'crf_fallback':{'folds':3,'search':'randomized, 10 configurations, seed 42',
                'trigger':'A training-only timing pilot predicts >4 CPU hours for the registered 5-fold search; log estimate and switch before viewing search scores.',
                'report':'Identify fallback explicitly; no switching based on CV/dev/test performance.'},
            'leakage_rule':'Fit vectorizers, learned vocabularies, scalers, SVD/PCA, K-means and all other learned transforms only on the fit portion of each fold. Refit selected transforms/model on train, never dev/test.',
            'features':'F3 gold POS/chunk is research-only; deployment variants use raw-text-available features.'},
        'transformer':{'encoders':['bert-base-cased','roberta-base'],'max_length':256,
            'preprocessing':'Accepted Step 3 first-subword supervision; reconstruct original sentences before scoring.',
            'selection':'Tune on validation with seed 42, freeze settings, then repeat the chosen setting for seeds 42/52/62. Each seed selects its checkpoint by validation F1.',
            'report':'Report all three seeds and mean/sample std (ddof=1). Never select the best test seed.',
            'primary_comparison_seed':42,'full_kfold':False,
            'reason':'Full K-fold fine-tuning is too costly; held-out development selection plus independent seed repeats.'},
        'uncertainty':{'n_resamples':2000,'confidence':0.95,'seed':BOOTSTRAP_SEED,'method':'sentence percentile',
            'paired_rule':'Same original sentence indices for A and B; resample saved predictions, no refitting.',
            'preplanned_primary_pairs':['selected deployable CRF vs BERT seed 42','BERT seed 42 vs RoBERTa seed 42'],
            'report':'B-minus-A intervals on corpus micro F1. Per-type comparisons exploratory; intervals are not multiple-comparison corrected or training-variance intervals.'},
        'change_control':'Never overwrite this v1 contract. Any necessary design change gets a new version and rationale before model selection/test scoring; keep the old contract. Register exact search spaces/budgets before search and final model settings before test.',
        'step4_scope':'Real data integrity/fold generation only. Metric/bootstrap/prediction-file demonstrations are synthetic. No real model predictions or test scores are generated.'}


def protocol_table(protocol):
    rows=[]
    for split,rule in protocol['split_usage'].items():
        rows.append(dict(component='data',setting=split,value=str(protocol['dataset']['audit']['splits'][split]['rows']),rule=rule))
    for key,value in protocol['seeds'].items():
        rows.append(dict(component='seed',setting=key,value=str(value),rule='Fixed before fitting/selection.'))
    for name,rule,value in [
        ('metric',METRIC_CONTRACT['primary'],'strict entity micro P/R/F1'),
        ('prediction_decode',METRIC_CONTRACT['prediction'],'orphan I-X -> B-X'),
        ('classical_cv',protocol['classical']['selection'],'5 contiguous document-safe folds'),
        ('crf_fallback',protocol['classical']['crf_fallback']['trigger'],'3 folds, randomized 10 configurations'),
        ('transform_fit',protocol['classical']['leakage_rule'],'fit inside each training fold'),
        ('transformer',protocol['transformer']['selection'],'3 seeds; validation selection'),
        ('bootstrap',METRIC_CONTRACT['bootstrap'],'2000 draws, 95%, seed 2026'),
        ('paired',METRIC_CONTRACT['paired'],'same resampled sentences'),
        ('test',protocol['split_usage']['test'],'once per frozen final checkpoint')]:
        rows.append(dict(component='protocol',setting=name,value=value,rule=rule))
    return rows


def freeze_protocol(project_root,protocol,fold_plans):
    directory=Path(project_root)/'config'/PROTOCOL_VERSION
    payloads={'protocol.json':canonical_bytes(protocol), **{name+'.json':canonical_bytes(plan) for name,plan in fold_plans.items()}}
    if directory.exists():
        if {p.name for p in directory.iterdir()}!=set(payloads) or any((directory/name).read_bytes()!=content for name,content in payloads.items()):
            raise ValueError('A different/incomplete v1 protocol already exists. Keep it; investigate or create an explicitly versioned revision.')
        reused=True
    else:
        directory.mkdir(parents=True,exist_ok=False)
        for name,content in payloads.items():
            with (directory/name).open('xb') as stream: stream.write(content)
        reused=False
    if any((directory/name).read_bytes()!=content for name,content in payloads.items()):
        raise ValueError('Frozen protocol read-back mismatch.')
    return dict(path=str(directory),protocol_sha256=fingerprint(protocol),reused_identical=reused,
        files=[dict(path=name,size_bytes=len(content),sha256=hashlib.sha256(content).hexdigest()) for name,content in payloads.items()],read_back_verified=True)

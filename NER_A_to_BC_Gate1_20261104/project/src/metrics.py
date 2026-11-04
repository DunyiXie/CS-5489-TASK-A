"""Versioned word-level NER evaluation. No training or data downloads."""
import math
from numbers import Integral

TYPES = ('PER', 'ORG', 'LOC', 'MISC')
LABELS = {'O'} | {f'{prefix}-{kind}' for prefix in ('B', 'I') for kind in TYPES}
METRIC_CONTRACT = {
    'version': 'ner-entity-metrics-v1',
    'primary': 'Micro entity precision/recall/F1; exact start, exclusive end and type, within each original sentence.',
    'seqeval': '1.2.2, mode=strict, scheme=IOB2, zero_division=0; independently checked against span counts.',
    'gold': 'Original metadata-decoded IOB2; invalid gold or mismatched sentence/token lengths raises an error.',
    'prediction': 'Before all scoring, an I-X not preceded by B-X/I-X becomes B-X. Other tags unchanged. Same fixed rule for every system; preserve raw tags and count repairs.',
    'rationale': 'Matches development-document Step 9 orphan-I extraction rule; does not inspect gold labels. Boundary/type matching remains exact.',
    'missing_types': 'Always output PER/ORG/LOC/MISC; undefined precision, recall or F1 is 0, with explicit support and denominator counts.',
    'bootstrap': '2000 sentence-level draws with replacement, 95% percentile intervals, NumPy default_rng seed 2026; sum TP/FP/FN then compute corpus metrics.',
    'paired': 'B minus A on identical sentence indices per draw; require identical ids, tokens and gold. An interval excluding zero is descriptive evidence, not a calibrated p-value.',
    'uncertainty_scope': 'Conditional on fixed trained models and sentence resampling; does not include training-seed variance or within-document dependence.',
    'scale': 'All metrics and confidence limits are fractions in [0,1]; paired differences in [-1,1].',
}


def _sentences(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(name + ' must be a nonempty sequence of original sentences.')
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row or any(not isinstance(t, str) or t not in LABELS for t in row):
            raise ValueError(name + ' contains an empty sentence, non-string or unknown tag.')
        result.append(list(row))
    return result


def entity_spans(tags):
    """Strict IOB2 -> set of (start, exclusive_end, type). Never crosses sentences."""
    if not isinstance(tags, (list, tuple)) or any(not isinstance(t, str) or t not in LABELS for t in tags):
        raise ValueError('Expected known IOB2 tag strings.')
    spans, start, kind = set(), None, None
    for i, tag in enumerate([*tags, 'O']):
        if tag.startswith('I-'):
            if start is None or kind != tag[2:]:
                raise ValueError('Invalid IOB2 continuation in gold/canonical tags.')
        else:
            if start is not None:
                spans.add((start, i, kind))
                start = kind = None
            if tag.startswith('B-'):
                start, kind = i, tag[2:]
    return spans


def canonicalize_predictions(y_pred):
    rows = _sentences(y_pred, 'predictions')
    repairs = 0
    output = []
    for row in rows:
        previous, canonical = 'O', []
        for tag in row:
            if tag.startswith('I-') and previous not in ('B-' + tag[2:], tag):
                tag = 'B-' + tag[2:]
                repairs += 1
            canonical.append(tag)
            previous = tag
        output.append(canonical)
    return output, repairs


def sentence_counts(y_true, y_pred):
    """Return counts[N, 4, 3] for TP/FP/FN, plus validated/canonical tag rows."""
    import numpy as np
    gold = _sentences(y_true, 'gold')
    pred, repairs = canonicalize_predictions(y_pred)
    if len(gold) != len(pred) or any(len(a) != len(b) for a,b in zip(gold,pred)):
        raise ValueError('Gold/prediction sentence or word counts differ.')
    counts = np.zeros((len(gold), len(TYPES), 3), dtype=np.int64)
    for i, (a,b) in enumerate(zip(gold,pred)):
        truth, guess = entity_spans(a), entity_spans(b)
        for j, kind in enumerate(TYPES):
            actual = {span for span in truth if span[2] == kind}
            found = {span for span in guess if span[2] == kind}
            counts[i,j] = (len(actual & found), len(found-actual), len(actual-found))
    return counts, gold, pred, repairs


def _scores(counts):
    import numpy as np
    counts = np.asarray(counts)
    tp, fp, fn = (counts[...,i] for i in range(3))
    def divide(a,b):
        return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    return np.stack((divide(tp,tp+fp), divide(tp,tp+fn), divide(2*tp,2*tp+fp+fn)),axis=-1)


def _count_report(counts):
    tp, fp, fn = map(int, counts)
    p,r,f = map(float, _scores(counts))
    return dict(tp=tp, fp=fp, fn=fn, support=tp+fn, predicted=tp+fp, precision=p, recall=r, f1=f)


def evaluate_ner(y_true, y_pred):
    """Primary project scorer: strict seqeval on fixed, gold-independent IOB2 decoding."""
    import numpy as np
    from seqeval.metrics import precision_score, recall_score, f1_score
    from seqeval.scheme import IOB2
    counts, gold, pred, repairs = sentence_counts(y_true,y_pred)
    totals = counts.sum(axis=0)
    overall = _count_report(totals.sum(axis=0))
    per_type = {kind:_count_report(totals[i]) for i,kind in enumerate(TYPES)}
    # Filter by type to verify every metric even when one type has no support.
    for kind, target in [('overall', overall), *per_type.items()]:
        true_rows = gold if kind=='overall' else [[t if t=='O' or t[2:]==kind else 'O' for t in row] for row in gold]
        pred_rows = pred if kind=='overall' else [[t if t=='O' or t[2:]==kind else 'O' for t in row] for row in pred]
        observed = [fn(true_rows,pred_rows,average='micro',mode='strict',scheme=IOB2,zero_division=0)
                    for fn in (precision_score,recall_score,f1_score)]
        if not np.allclose(observed,[target[k] for k in ('precision','recall','f1')],atol=1e-12,rtol=0):
            raise ValueError('seqeval differs from independent exact-span counts: ' + kind)
    return dict(contract=METRIC_CONTRACT['version'],overall=overall,per_type=per_type,
                sentences=len(gold),words=sum(map(len,gold)),prediction_iob2_repairs=repairs,
                seqeval_crosscheck_passed=True)


def entity_f1(y_true,y_pred):
    """CV scoring function; call with lists of original sentence label lists."""
    return evaluate_ner(y_true,y_pred)['overall']['f1']


def estimator_entity_f1(estimator, X, y_true):
    """scikit-learn callable scorer. Estimator.predict(X) must return sentence lists."""
    def nested_lists(value):
        rows=list(value)
        if any(isinstance(row,str) for row in rows):
            raise ValueError('CV scorer requires sentence lists, not flattened token labels.')
        return [list(row) for row in rows]
    return entity_f1(nested_lists(y_true),nested_lists(estimator.predict(X)))


def _bootstrap_args(n_resamples,confidence,seed):
    if isinstance(n_resamples,bool) or not isinstance(n_resamples,Integral) or n_resamples<2:
        raise ValueError('n_resamples must be an integer >= 2.')
    if isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0:
        raise ValueError('seed must be a nonnegative integer.')
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<confidence<1:
        raise ValueError('confidence must be between zero and one.')


def _with_overall(counts):
    import numpy as np
    return np.concatenate((counts.sum(axis=1,keepdims=True),counts),axis=1)


def _intervals(samples, point, confidence):
    import numpy as np
    alpha=(1-confidence)/2
    limits=np.quantile(samples,[alpha,1-alpha],axis=0,method='linear')
    return {kind:{metric:dict(estimate=float(point[i,j]),low=float(limits[0,i,j]),high=float(limits[1,i,j]))
             for j,metric in enumerate(('precision','recall','f1'))}
            for i,kind in enumerate(('overall',*TYPES))}


def bootstrap_ci(y_true,y_pred,n_resamples=2000,confidence=0.95,seed=2026):
    """Sentence percentile bootstrap; resample counts, not token rows or sentence F1s."""
    import numpy as np
    _bootstrap_args(n_resamples,confidence,seed)
    point_report=evaluate_ner(y_true,y_pred)
    counts=_with_overall(sentence_counts(y_true,y_pred)[0])
    rng=np.random.default_rng(seed)
    samples=np.empty((n_resamples,5,3)); empty=np.zeros(5,dtype=np.int64)
    for b in range(n_resamples):
        total=counts[rng.integers(0,len(counts),size=len(counts))].sum(axis=0)
        samples[b]=_scores(total)
        empty += (total.sum(axis=1)==0)
    return dict(method='sentence_percentile',seed=int(seed),n_resamples=int(n_resamples),confidence=confidence,
        unit='original sentence',point=point_report,intervals=_intervals(samples,_scores(counts.sum(axis=0)),confidence),
        zero_f1_denominator_draws=dict(zip(('overall',*TYPES),map(int,empty))),
        limitation=METRIC_CONTRACT['uncertainty_scope'])


def paired_bootstrap(y_true,y_pred_a,y_pred_b,n_resamples=2000,confidence=0.95,seed=2026):
    """B-minus-A percentile intervals using exactly the same indices in each draw.

    Arrays must already be aligned to the same original sentences. For saved files,
    use prediction_io.compare_prediction_files to validate ids/tokens/gold first.
    """
    import numpy as np
    _bootstrap_args(n_resamples,confidence,seed)
    point_a,point_b=evaluate_ner(y_true,y_pred_a),evaluate_ner(y_true,y_pred_b)
    a=_with_overall(sentence_counts(y_true,y_pred_a)[0]); b=_with_overall(sentence_counts(y_true,y_pred_b)[0])
    rng=np.random.default_rng(seed); samples=np.empty((n_resamples,5,3))
    for i in range(n_resamples):
        selected=rng.integers(0,len(a),size=len(a))
        samples[i]=_scores(b[selected].sum(axis=0))-_scores(a[selected].sum(axis=0))
    delta=_scores(b.sum(axis=0))-_scores(a.sum(axis=0))
    intervals=_intervals(samples,delta,confidence)
    return dict(method='paired_sentence_percentile',direction='B minus A',seed=int(seed),n_resamples=int(n_resamples),
        confidence=confidence,point_a=point_a,point_b=point_b,intervals=intervals,
        overall_f1_interval_excludes_zero=bool(intervals['overall']['f1']['low']>0 or intervals['overall']['f1']['high']<0),
        interpretation='Percentile difference interval; no p-value computed. Predeclare comparisons; per-type intervals are not multiplicity-adjusted.',
        limitation=METRIC_CONTRACT['uncertainty_scope'])


def toy_checks():
    """Real seqeval checks on fabricated tags only; not model performance."""
    cases=[
        ('exact',[['B-PER','I-PER','O']],[['B-PER','I-PER','O']],(1,0,0)),
        ('Jane_only',[['B-PER','I-PER']],[['B-PER','O']],(0,1,1)),
        ('wrong_type',[['B-PER','I-PER']],[['B-ORG','I-ORG']],(0,1,1)),
        ('missed',[['B-LOC']],[['O']],(0,0,1)),
        ('spurious',[['O']],[['B-MISC']],(0,1,0)),
        ('all_O',[['O','O']],[['O','O']],(0,0,0)),
        ('adjacent_entities',[['B-PER','B-PER']],[['B-PER','B-PER']],(2,0,0)),
        ('sentence_boundary',[['B-PER'],['B-PER']],[['B-PER'],['I-PER']],(2,0,0)),
        ('orphan_I_false_positive',[['O']],[['I-PER']],(0,1,0)),
        ('type_switch_I',[['B-PER','B-ORG']],[['B-PER','I-ORG']],(2,0,0)),
    ]
    results=[]
    for name,gold,pred,expected in cases:
        result=evaluate_ner(gold,pred)
        if tuple(result['overall'][k] for k in ('tp','fp','fn'))!=expected:
            raise ValueError('Toy metric mismatch: '+name)
        results.append(dict(case=name,synthetic_only=True,**result))
    return results

"""Descriptive NER statistics. No training, model selection or test-set scoring."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ('train', 'validation', 'test')
TYPES = ('PER', 'ORG', 'LOC', 'MISC')
PAPER = {
    'train': dict(documents=946, paper_sentences=14987, sentences=14041, tokens=203621,
                  LOC=7140, MISC=3438, ORG=6321, PER=6600),
    'validation': dict(documents=216, paper_sentences=3466, sentences=3250, tokens=51362,
                       LOC=1837, MISC=922, ORG=1341, PER=1842),
    'test': dict(documents=231, paper_sentences=3684, sentences=3453, tokens=46435,
                 LOC=1668, MISC=702, ORG=1661, PER=1617),
}
DEFINITIONS = {
    'entity': 'Strict IOB2 span, counted once at B-. End index is exclusive.',
    'surface': 'Exact tuple of original entity tokens; case-sensitive, no punctuation or whitespace normalization.',
    'unseen_primary': 'Surface absent among ALL training gold entity types. Mention rate weights occurrences; unique rate weights distinct surfaces within each type.',
    'unseen_typed': 'Surface absent from training gold entities of the SAME type; reported separately.',
    'casefold': 'Supplemental case-insensitive comparison uses str.casefold() on each token; original data are unchanged.',
    'capitalization': 'First character of the original first token is uppercase (str.isupper). Entity denominator: all mentions of that type. Sentence-O denominator: sentences whose token at index 0 has O; no punctuation skipping.',
    'transition': 'Only adjacent tokens within a sentence; no transitions across sentences. Probability divides by outgoing transitions for that label. Zero denominator is null (N/A).',
    'ambiguity': 'Exact token forms or complete entity surfaces with at least two distinct entity types WITHIN a split. O is excluded from entity types; O occurrence counts are supplemental.',
    'vocabulary': 'Includes punctuation and digits. Directed source-to-target type coverage = intersection / target vocabulary size; token coverage weights target token occurrences. Jaccard = intersection / union.',
    'top_entities': 'Top 10 per split and type, by mention count descending then token tuple ascending; plots show training entities.',
    'paper_sentences': 'Compare HF sentences directly to 14041/3250/3453; also compare HF sentences + document count to paper sentence count. DOCSTART explanation is an inference, not a reconstruction of the original files.',
    'test_use': 'Test statistics are descriptive only and must not select features, hyperparameters, preprocessing or models. No prediction scores are computed.',
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def extract_entities(tokens, tags):
    """Return (start, end, type, token_tuple), rejecting illegal IOB2."""
    if len(tokens) != len(tags):
        raise ValueError('Token/tag length mismatch.')
    spans = []
    start = kind = None
    for index, tag in enumerate([*tags, 'O']):
        if tag != 'O' and (tag[:2] not in ('B-', 'I-') or tag[2:] not in TYPES):
            raise ValueError(f'Unknown tag: {tag}')
        if tag.startswith('I-'):
            if start is None or kind != tag[2:]:
                raise ValueError('Illegal IOB2 continuation.')
        else:
            if start is not None:
                spans.append((start, index, kind, tuple(tokens[start:index])))
                start = kind = None
            if tag.startswith('B-'):
                start, kind = index, tag[2:]
    return spans


def _normal_surface(surface, mode):
    return surface if mode == 'exact' else tuple(token.casefold() for token in surface)


def analyze(dataset, labels):
    """One deterministic traversal; keeps distributions/aggregates rather than full rows."""
    tables = {name: [] for name in ('summary', 'labels', 'sentence_lengths', 'entity_lengths',
              'top_entities', 'transitions', 'capitalization', 'unseen', 'vocabulary',
              'ambiguous_tokens', 'ambiguous_entities', 'order_spotchecks', 'paper_comparison')}
    entities_by_split, vocab_by_split = {}, {}
    for split in SPLITS:
        rows = dataset[split]
        label_counts, lengths, transitions, vocab = Counter(), Counter(), Counter(), Counter()
        entity_counts = {kind: Counter() for kind in TYPES}
        entity_lengths = {kind: Counter() for kind in TYPES}
        caps = Counter()
        token_types, surface_types = defaultdict(Counter), defaultdict(Counter)
        outside_tokens = Counter()
        docs, spots = set(), set(range(min(3, len(rows))))
        middle = len(rows) // 2
        spots.update(range(max(0, middle - 1), min(len(rows), middle + 2)))
        previous_doc = None
        boundary_found = False
        for index, row in enumerate(rows):
            tokens = row['tokens']
            tags = [labels[tag] for tag in row['ner_tags']]
            spans = extract_entities(tokens, tags)
            label_counts.update(tags)
            lengths[len(tokens)] += 1
            vocab.update(tokens)
            transitions.update(zip(tags, tags[1:]))
            docs.add(row['document_id'])
            if previous_doc is not None and row['document_id'] != previous_doc and not boundary_found:
                spots.update([index - 1, index])
                boundary_found = True
            previous_doc = row['document_id']
            if tags and tags[0] == 'O':
                caps['sentence_O_total'] += 1
                caps['sentence_O_upper'] += int(tokens[0][:1].isupper())
            for token, tag in zip(tokens, tags):
                if tag == 'O':
                    outside_tokens[token] += 1
                else:
                    token_types[token][tag[2:]] += 1
            for start, end, kind, surface in spans:
                entity_counts[kind][surface] += 1
                entity_lengths[kind][end - start] += 1
                surface_types[surface][kind] += 1
                caps[kind] += int(surface[0][:1].isupper())
            if len(spans) != sum(tag.startswith('B-') for tag in tags):
                raise ValueError('Span/B-tag count mismatch.')
        total_tokens = sum(vocab.values())
        counts = {kind: sum(entity_counts[kind].values()) for kind in TYPES}
        if sum(label_counts.values()) != total_tokens or sum(transitions.values()) != total_tokens - len(rows):
            raise ValueError('Count/transition conservation failed.')
        summary = dict(split=split, sentences=len(rows), tokens=total_tokens, documents=len(docs),
                       entities=sum(counts.values()), **counts)
        tables['summary'].append(summary)
        for metric in ('sentences', 'tokens', 'documents', *TYPES):
            expected = PAPER[split][metric]
            tables['paper_comparison'].append(dict(split=split, metric=metric,
                observed=summary[metric], reference=expected, difference=summary[metric] - expected))
        adjusted = len(rows) + len(docs)
        tables['paper_comparison'].append(dict(split=split, metric='sentences_plus_documents',
            observed=adjusted, reference=PAPER[split]['paper_sentences'],
            difference=adjusted - PAPER[split]['paper_sentences']))
        tables['paper_comparison'].append(dict(split=split, metric='raw_sentences_vs_paper',
            observed=len(rows), reference=PAPER[split]['paper_sentences'],
            difference=len(rows) - PAPER[split]['paper_sentences'], expected_difference=-PAPER[split]['documents']))
        for label in labels:
            tables['labels'].append(dict(split=split, label=label, count=label_counts[label],
                                        proportion=ratio(label_counts[label], total_tokens)))
            outgoing = sum(transitions[(label, other)] for other in labels)
            for other in labels:
                n = transitions[(label, other)]
                tables['transitions'].append(dict(split=split, previous=label, following=other,
                                                  count=n, probability=ratio(n, outgoing)))
        for length, count in sorted(lengths.items()):
            tables['sentence_lengths'].append(dict(split=split, length=length, count=count))
        for kind in TYPES:
            for length, count in sorted(entity_lengths[kind].items()):
                tables['entity_lengths'].append(dict(split=split, type=kind, length=length,
                                                     count=count, proportion=ratio(count, counts[kind])))
            for rank, (surface, count) in enumerate(sorted(entity_counts[kind].items(),
                                                           key=lambda x: (-x[1], x[0]))[:10], 1):
                tables['top_entities'].append(dict(split=split, type=kind, rank=rank,
                                                  surface=' '.join(surface), count=count))
            tables['capitalization'].append(dict(split=split, metric=kind, uppercase=caps[kind],
                                                 total=counts[kind], proportion=ratio(caps[kind], counts[kind])))
        tables['capitalization'].append(dict(split=split, metric='sentence_initial_O',
            uppercase=caps['sentence_O_upper'], total=caps['sentence_O_total'],
            proportion=ratio(caps['sentence_O_upper'], caps['sentence_O_total'])))
        for table, forms in [('ambiguous_tokens', token_types), ('ambiguous_entities', surface_types)]:
            ambiguous = [(form, values) for form, values in forms.items() if len(values) > 1]
            for form, values in sorted(ambiguous, key=lambda x: (-sum(x[1].values()), x[0])):
                item = dict(split=split, surface=form if table == 'ambiguous_tokens' else ' '.join(form),
                            types='|'.join(sorted(values)), total=sum(values.values()),
                            **{kind: values[kind] for kind in TYPES})
                if table == 'ambiguous_tokens':
                    item['O_count'] = outside_tokens[form]
                tables[table].append(item)
        for index in sorted(spots):
            row = rows[index]
            tables['order_spotchecks'].append(dict(split=split, row=index, id=row['id'],
                document_id=row['document_id'], sentence_id=row['sentence_id'],
                preview=' '.join(row['tokens'][:24])))
        entities_by_split[split], vocab_by_split[split] = entity_counts, vocab

    for mode in ('exact', 'casefold'):
        train_by_type = {kind: {_normal_surface(s, mode) for s in entities_by_split['train'][kind]}
                         for kind in TYPES}
        train_any = set().union(*train_by_type.values())
        for split in ('validation', 'test'):
            for kind in TYPES:
                target = Counter()
                for surface, count in entities_by_split[split][kind].items():
                    target[_normal_surface(surface, mode)] += count
                unseen = {s for s in target if s not in train_any}
                unseen_typed = {s for s in target if s not in train_by_type[kind]}
                n, unique = sum(target.values()), len(target)
                tables['unseen'].append(dict(split=split, normalization=mode, type=kind,
                    mentions=n, unique_surfaces=unique,
                    unseen_mentions=sum(target[s] for s in unseen), unseen_unique=len(unseen),
                    mention_rate=ratio(sum(target[s] for s in unseen), n), unique_rate=ratio(len(unseen), unique),
                    unseen_typed_mentions=sum(target[s] for s in unseen_typed), unseen_typed_unique=len(unseen_typed),
                    typed_mention_rate=ratio(sum(target[s] for s in unseen_typed), n),
                    typed_unique_rate=ratio(len(unseen_typed), unique)))
        vocabs = {}
        for split in SPLITS:
            values = Counter()
            for token, count in vocab_by_split[split].items():
                values[token if mode == 'exact' else token.casefold()] += count
            vocabs[split] = values
        for source in SPLITS:
            for target in SPLITS:
                a, b = set(vocabs[source]), set(vocabs[target])
                overlap = a & b
                covered = sum(vocabs[target][t] for t in overlap)
                tables['vocabulary'].append(dict(normalization=mode, source=source, target=target,
                    source_types=len(a), target_types=len(b), shared_types=len(overlap), union_types=len(a | b),
                    type_coverage=ratio(len(overlap), len(b)), jaccard=ratio(len(overlap), len(a | b)),
                    covered_target_tokens=covered, target_tokens=sum(vocabs[target].values()),
                    token_coverage=ratio(covered, sum(vocabs[target].values()))))
    for row in tables['paper_comparison']:
        row.setdefault('expected_difference', 0)
        row['matches_expected'] = row['difference'] == row['expected_difference']
    return {'definitions': DEFINITIONS, 'labels_from_metadata': list(labels), 'tables': tables,
            'paper_comparison_passed': all(row['matches_expected'] for row in tables['paper_comparison'])}


def ensure_dataset(project_root, accepted, run_id):
    """Reuse the accepted Step 1 files, or redownload exactly the same pinned content."""
    import step1_data as data_tools
    from datasets import load_from_disk

    project_root = Path(project_root)
    path = project_root / 'data' / Path(accepted['saved_data']['path']).name
    if path.is_dir():
        for record in accepted['saved_data']['files']:
            file = path / record['path']
            if not file.is_file() or data_tools.file_sha256(file) != record['sha256']:
                raise ValueError('Saved Step 1 data changed. Stop and report this error.')
        dataset = load_from_disk(str(path))
        source = accepted['source']
        recovery = 'reused-verified-step1-files'
    else:
        dataset, source = data_tools.load_pinned_dataset(project_root)
        if source['source_files'] != accepted['source']['source_files']:
            raise ValueError('Pinned downloads differ from the accepted Step 1 source hashes.')
        recovery = 'redownloaded-pinned-step1-source'
    audit = data_tools.audit_dataset(dataset)
    if audit != accepted['audit']:
        raise ValueError('Dataset differs from the accepted Step 1 ordered audit.')
    if recovery.startswith('redownloaded'):
        saved = data_tools.save_and_verify(dataset, project_root, run_id, audit)
    else:
        saved = {**accepted['saved_data'], 'path': str(path)}
    record = dict(source=source, saved_data=saved, audit=audit, recovery=recovery, run_id=run_id)
    data_tools.write_json(project_root / 'config' / 'dataset_latest.json', record)
    return dataset, record


def write_tables(report, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in report['tables'].items():
        path = destination / f'{name}.csv'
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            fields = list(rows[0]) if rows else ['split', 'surface', 'types', 'total', *TYPES]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (destination / 'eda_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Step 2 descriptive EDA', '', 'No model training or prediction scoring.', '',
             'Reference: https://aclanthology.org/W03-0419.pdf (Tables 1 and 2).', '', '## Definitions', '']
    lines.extend(f'- {key}: {value}' for key, value in report['definitions'].items())
    lines += ['', '## Observed counts', '', '| Split | Sentences | Tokens | Entities | PER | ORG | LOC | MISC |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['tables']['summary']:
        lines.append('| ' + ' | '.join(str(row[k]) for k in ('split', 'sentences', 'tokens', 'entities', *TYPES)) + ' |')
    lines += ['', f"Paper count comparison passed: {report['paper_comparison_passed']}.",
              'Raw paper sentence counts include an additional document-sized difference; see paper_comparison.csv.',
              'Use train-only observations when considering features. Test statistics are descriptive, not a selection criterion.',
              'PNG and vector PDF figures are stored in the corresponding figures/step2_<run_id> directory.']
    (destination / 'EDA_README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


FIGURE_NAMES = ('01_label_distribution', '02_sentence_lengths', '03_entity_lengths',
                '04_top_entities_train', '05_tag_transitions', '06_capitalization',
                '07_unseen_entities', '08_vocabulary_overlap', '09_ambiguity_train')


def plot_report(report, destination):
    """Nine report figures, each exported as a PNG and a vector PDF."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    from textwrap import fill

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables, labels = report['tables'], report['labels_from_metadata']
    colors = ['#2670AA', '#D17B22', '#36846B', '#A75085']
    paths = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
             'axes.spines.right': False, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'}

    def selected(name, **where):
        return [r for r in tables[name] if all(r[k] == v for k, v in where.items())]

    def save(fig, name):
        fig.savefig(destination / (name + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(destination / (name + '.pdf'), bbox_inches='tight')
        paths.append(str(destination / (name + '.png')))
        plt.close(fig)

    def number(value):
        return float('nan') if value is None else value

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained', sharey=True)
        for ax, split, color in zip(axes, SPLITS, colors):
            rows = selected('labels', split=split)
            ax.bar([r['label'] for r in rows], [r['proportion'] for r in rows], color=color)
            ax.set(title=split, xlabel='Token label (includes O)')
            ax.tick_params(axis='x', rotation=55)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_ylabel('Share of tokens')
        fig.suptitle('NER label distribution')
        save(fig, FIGURE_NAMES[0])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
        maximum = max(r['length'] for r in tables['sentence_lengths'])
        bins = np.arange(0.5, maximum + 5.5, 5)
        for ax, split, color in zip(axes, SPLITS, colors):
            rows = selected('sentence_lengths', split=split)
            ax.hist([r['length'] for r in rows], bins=bins, weights=[r['count'] for r in rows], color=color)
            ax.set(title=f'{split} (max {max(r["length"] for r in rows)})', xlabel='Tokens per sentence', ylabel='Sentences', xlim=(0, maximum + 1))
        fig.suptitle('Sentence lengths — all sentences, no tail truncation')
        save(fig, FIGURE_NAMES[1])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        for ax, kind in zip(axes.flat, TYPES):
            kind_rows = selected('entity_lengths', type=kind)
            maximum = max((r['length'] for r in kind_rows), default=1)
            x = np.arange(1, maximum + 1)
            for split, color in zip(SPLITS, colors):
                rows = selected('entity_lengths', split=split, type=kind)
                counts = {r['length']: r['count'] for r in rows}
                total = sum(counts.values())
                probabilities = [counts.get(length, 0) / total if total else np.nan for length in x]
                ax.plot(x, probabilities, marker='o', ms=3, label=split, color=color)
            ax.set(title=kind, xlabel='Tokens per entity', ylabel='Share within split/type',
                   ylim=(0, 1.05), xlim=(.5, maximum + .5))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(fontsize=8)
        fig.suptitle('Entity length distributions')
        save(fig, FIGURE_NAMES[2])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
        for ax, kind, color in zip(axes.flat, TYPES, colors):
            rows = selected('top_entities', split='train', type=kind)
            ax.barh(range(len(rows)), [r['count'] for r in rows], color=color)
            ax.set_yticks(range(len(rows)), [fill(r['surface'], 26) for r in rows], fontsize=9)
            ax.invert_yaxis()
            ax.set(title=kind, xlabel='Entity mentions in training data')
        fig.suptitle('Ten most frequent entity surfaces per type (train only)')
        save(fig, FIGURE_NAMES[3])

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), layout='constrained')
        for ax, split in zip(axes, SPLITS):
            lookup = {(r['previous'], r['following']): r['probability'] for r in selected('transitions', split=split)}
            matrix = np.array([[number(lookup[(a, b)]) for b in labels] for a in labels])
            im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right')
            ax.set_yticks(range(len(labels)), labels)
            ax.set(title=split, xlabel='Next label', ylabel='Current label')
            for i in range(len(labels)):
                for j in range(len(labels)):
                    v = matrix[i, j]
                    ax.text(j, i, 'N/A' if np.isnan(v) else f'{100*v:.0f}', ha='center', va='center',
                            fontsize=7, color='white' if v > .55 else '#172B40')
        fig.colorbar(im, ax=axes, shrink=.65, label='Row probability')
        fig.suptitle('Within-sentence label transitions — cell values in percent')
        save(fig, FIGURE_NAMES[4])

        fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')
        categories = [*TYPES, 'sentence_initial_O']
        x = np.arange(len(categories))
        for i, (split, color) in enumerate(zip(SPLITS, colors)):
            lookup = {r['metric']: r for r in selected('capitalization', split=split)}
            ax.bar(x + (i-1)*.25, [number(lookup[k]['proportion']) for k in categories], width=.24, color=color, label=split)
        ax.set_xticks(x, [*TYPES, 'Sentence-first O token'])
        ax.set(ylabel='Uppercase first character / eligible occurrences', ylim=(0, 1.05),
               title='Capitalization of entity starts and sentence-initial non-entities')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        save(fig, FIGURE_NAMES[5])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained', sharey=True)
        x = np.arange(len(TYPES))
        for ax, rate, title in zip(axes, ('mention_rate', 'unique_rate'), ('By mention occurrence', 'By unique surface')):
            for i, (split, color) in enumerate(zip(('validation', 'test'), colors[1:])):
                lookup = {r['type']: r for r in selected('unseen', split=split, normalization='exact')}
                ax.bar(x + (i-.5)*.34, [number(lookup[k][rate]) for k in TYPES], width=.33, label=split, color=color)
            ax.set_xticks(x, TYPES)
            ax.set(title=title, ylabel='Unseen proportion', ylim=(0, 1.05))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend()
        fig.suptitle('Entity surfaces unseen among training entities (exact case, any type)')
        save(fig, FIGURE_NAMES[6])

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout='constrained')
        for ax, mode in zip(axes, ('exact', 'casefold')):
            lookup = {(r['source'], r['target']): r['type_coverage'] for r in selected('vocabulary', normalization=mode)}
            matrix = np.array([[number(lookup[(a, b)]) for b in SPLITS] for a in SPLITS])
            im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1)
            ax.set_xticks(range(3), SPLITS)
            ax.set_yticks(range(3), SPLITS)
            ax.set(title=mode, xlabel='Target vocabulary (denominator)', ylabel='Source vocabulary')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{matrix[i,j]:.1%}', ha='center', va='center', color='white' if matrix[i,j] > .65 else 'black')
        fig.colorbar(im, ax=axes, shrink=.75, label='Target type coverage')
        fig.suptitle('Directed vocabulary overlap (includes punctuation and digits)')
        save(fig, FIGURE_NAMES[7])

        fig, axes = plt.subplots(1, 2, figsize=(16, 10), layout='constrained')
        for ax, name, title in zip(axes, ('ambiguous_tokens', 'ambiguous_entities'), ('Token forms', 'Complete entity surfaces')):
            rows = selected(name, split='train')[:15]
            if not rows:
                ax.text(.5, .5, 'No multi-type forms', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue
            left = np.zeros(len(rows))
            for kind, color in zip(TYPES, colors):
                values = np.array([r[kind] for r in rows])
                ax.barh(range(len(rows)), values, left=left, label=kind, color=color)
                left += values
            ax.set_yticks(range(len(rows)), [fill(r['surface'], 25) for r in rows], fontsize=9)
            ax.invert_yaxis()
            ax.set(title=title, xlabel='Entity-token occurrences' if name == 'ambiguous_tokens' else 'Entity mentions')
            ax.legend(fontsize=8)
        fig.suptitle('Multi-type ambiguity in training data (up to 15 most frequent forms; O excluded)')
        save(fig, FIGURE_NAMES[8])
    return paths

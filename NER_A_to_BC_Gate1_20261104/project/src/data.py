"""Stable shared data entry points; source loading remains in step1_data.py."""
from metrics import LABELS


def label_mapping(dataset):
    from step1_data import label_names
    names=label_names(dataset['train'])['ner_tags']
    if set(names)!=LABELS or any(label_names(split)['ner_tags']!=names for split in dataset.values()):
        raise ValueError('Label metadata differs across splits or from the metric contract.')
    return dict(label_names=names,label2id={name:i for i,name in enumerate(names)},
                id2label={i:name for i,name in enumerate(names)},scheme='IOB2')


def load_verified_dataset(project_root,accepted_reference,run_id):
    from step2_eda import ensure_dataset
    dataset,record=ensure_dataset(project_root,accepted_reference,run_id)
    return dataset,record,label_mapping(dataset)


def decoded_gold(rows,label_names):
    """Only call on the authorized split; this function does not choose a split."""
    return [[label_names[v] for v in row['ner_tags']] for row in rows]

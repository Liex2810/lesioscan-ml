"""Patient-disjoint, class-aware holdout selection using repeated group candidates.

No cross-split patient overlap. Class balancing is heuristic, not a guarantee;
raises with actionable context if coverage is impossible.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _score(candidate, labels, proportions, classes):
    total = len(candidate)
    sizes = candidate['split'].value_counts(normalize=True)
    distribution = candidate[labels].value_counts(normalize=True)
    total_score = sum(abs(float(sizes.get(k, 0)) - v) for k,v in proportions.items())
    for split, fraction in proportions.items():
        piece = candidate[candidate['split'] == split]
        if piece.empty:
            return float('inf')
        counts = piece[labels].value_counts(normalize=True)
        total_score += 0.8 * sum(abs(float(counts.get(c, 0)) - float(distribution.get(c, 0))) for c in classes)
        missing = set(classes) - set(piece[labels])
        total_score += 50 * len(missing)
    return total_score


def stratified_patient_split(frame, patient_col='patient_id', label_col='diagnostic', holdout=0.15,
                             val=0.15, seed=42, n_candidates=150, require_all_classes=True):
    if not (0 < holdout < 1 and 0 < val < 1 and holdout + val < 1):
        raise ValueError('holdout/val must be between 0 and 1, with sum < 1')
    if n_candidates < 1:
        raise ValueError('n_candidates must be >= 1')
    df=frame.copy().reset_index(drop=True)
    if df[[patient_col, label_col]].isna().any().any():
        raise ValueError('Missing patient IDs or labels; clean data first')
    df[patient_col]=df[patient_col].astype(str)
    classes=sorted(df[label_col].astype(str).unique())
    counts=df.groupby(label_col)[patient_col].nunique()
    if require_all_classes and (counts < 3).any():
        raise ValueError('Each class needs >=3 distinct patients to appear in train, val and test: '+str(counts.to_dict()))
    proportions={'train':1-holdout-val,'val':val,'test':holdout}
    best=None; best_score=float('inf')
    group=df[patient_col]
    for i in range(n_candidates):
        first=GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed+2*i)
        remain_idx,test_idx=next(first.split(df,groups=group))
        remaining=df.iloc[remain_idx]
        second=GroupShuffleSplit(n_splits=1,test_size=val/(1-holdout),random_state=seed+2*i+1)
        tr_idx,va_idx=next(second.split(remaining,groups=remaining[patient_col]))
        candidate=df.copy()
        candidate['split']='test'
        candidate.loc[remaining.iloc[tr_idx].index,'split']='train'
        candidate.loc[remaining.iloc[va_idx].index,'split']='val'
        score=_score(candidate,label_col,proportions,classes)
        if score < best_score:
            best=candidate;best_score=score
    if best is None:
        raise RuntimeError('Unable to generate a nonempty patient-level split')
    groups=[set(best.loc[best.split==name,patient_col]) for name in ('train','val','test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)):
        raise RuntimeError('Patient leakage detected')
    coverage={s: sorted(set(classes)-set(best.loc[best.split==s,label_col])) for s in ('train','val','test')}
    if require_all_classes and any(coverage.values()):
        raise ValueError(f'No candidate covered all classes after {n_candidates} attempts: {coverage}. '
                         'Increase split_candidates, inspect patient-level class distribution, or revise split protocol.')
    return best

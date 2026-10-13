"""
Module: experiments/research_v2/sealed_guard.py
Description: Cryptographic and runtime assertion guard ensuring the 33 sealed cohort
participants are NEVER loaded, fitted on, scaled, tuned, or evaluated outside Phase 6.
"""

import os
import json
import re
from typing import Iterable, Set, Union, List
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SEALED_PATH = os.path.join(_REPO_ROOT, 'sealed_ids.json')

def load_sealed_pids() -> Set[int]:
    if not os.path.exists(_SEALED_PATH):
        raise FileNotFoundError(f'Sealed cohort ID specification not found at {_SEALED_PATH}')
    with open(_SEALED_PATH, 'r') as f:
        data = json.load(f)
    return set(int(p) for p in data['sealed_pids'])

SEALED_PIDS: Set[int] = load_sealed_pids()

def extract_pid(val: Union[int, str]) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    val_str = str(val).strip()
    match = re.match(r'^([0-9]+)', val_str)
    if match:
        return int(match.group(1))
    raise ValueError(f'Cannot parse participant ID from value: {val}')

def assert_no_sealed_ids(
    ids: Iterable[Union[int, str]],
    context: str = 'data_loading',
    allow_sealed: bool = False
) -> None:
    """
    Enforces zero overlap with sealed cohort.
    Raises RuntimeError immediately if any sealed participant is detected outside Phase 6.
    """
    if allow_sealed:
        return

    detected_sealed = set()
    for item in ids:
        try:
            pid = extract_pid(item)
            if pid in SEALED_PIDS:
                detected_sealed.add(pid)
        except ValueError:
            pass

    if detected_sealed:
        raise RuntimeError(
            f'CRITICAL DATA LEAKAGE VIOLATION in {context}! '
            f'Found {len(detected_sealed)} sealed cohort participants: {sorted(detected_sealed)}. '
            f'The sealed cohort (N=33) is strictly quarantined and must NEVER be loaded, fitted, '
            f'scaled, tuned, or evaluated prior to Phase 6.'
        )

def guard_dataframe(
    df: pd.DataFrame,
    id_col: str = 'participant_id',
    context: str = 'dataframe_loading',
    allow_sealed: bool = False
) -> pd.DataFrame:
    """
    Validates a DataFrame and raises if sealed IDs are present outside Phase 6.
    """
    if id_col in df.columns:
        assert_no_sealed_ids(df[id_col].values, context=context, allow_sealed=allow_sealed)
    return df

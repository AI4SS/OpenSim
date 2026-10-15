"""Environment posts, event schedule, and count/state feedback."""
from copy import deepcopy
import json
from pathlib import Path

from .population import content_type, record_text, source_day
from .state import update_shadow_state


def load_event(micro_path, macro_path):
    records = json.loads(Path(micro_path).read_text(encoding='utf-8'))
    with Path(macro_path).open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not isinstance(records, list) or not records or not rows:
        raise ValueError('Micro records and macro timeline must be nonempty')
    interactions = [r for r in records[1:] if content_type(r) != 'post']
    if not interactions:
        raise ValueError('The event has no reference actors')
    by_day = {}
    for record in records[1:]:
        by_day.setdefault(source_day(record), []).append(record)
    return records[0], rows, interactions, by_day


def day_root(root, day_posts):
    texts = [record_text(r).strip() for r in day_posts[:5] if record_text(r).strip()]
    if not texts:
        return root
    result = deepcopy(root)
    result['text'] = f"{root.get('text', '')}\n\n当日主帖/干扰项环境输入：" + '；'.join(texts)
    return result


def update_history(rows, day_idx, frame, cumulative_state, summary, scale_factor,
                   *, state_history_volume='simulated'):
    """Write completed-day feedback in full sample-count units for later days."""
    if state_history_volume not in {'observed', 'simulated'}:
        raise ValueError('State history volume must be observed or simulated')
    count = len(frame) * scale_factor
    if state_history_volume == 'simulated':
        rows[day_idx].setdefault('output', {})['num_interactions'] = count
    actions = frame['generated_action'].astype(str).str.lower()
    share_ratio = float((actions == 'share').mean())
    diagnostics = rows[day_idx].setdefault('diagnostics', {})
    diagnostics.update(sample_num_interactors=count,
                       num_comments=round(count * (1 - share_ratio)),
                       num_shares=round(count * share_ratio),
                       comment_share_ratio={'comment': 1 - share_ratio, 'share': share_ratio},
                       weighted_interactions={'comment_ratio': 1 - share_ratio, 'share_ratio': share_ratio})
    import numpy as np
    daily = np.mean(np.stack(frame['individual_mu_dist'].tolist()), axis=0)
    update_shadow_state(rows=rows, day_idx=day_idx, daily_dist=daily,
                        cumulative_dist=cumulative_state, mean_field_summary=summary)

"""Reference actors sampled with replacement from the complete event pool."""
from copy import deepcopy
import random
import re


def record_text(record):
    return str(record.get('text') or record.get('content') or record.get('original_text') or record.get('source_text') or '')


def content_type(record):
    value = str(record.get('source_content_type') or record.get('content_type') or '').strip().lower()
    text = record_text(record).strip()
    if value == 'share' or text in {'转发微博', '转发'}:
        return 'share'
    if value == 'post' or str(record.get('disturb', '')).strip() == '1':
        return 'post'
    return 'comment'


def source_day(record):
    value = record.get('t') or record.get('source_t') or record.get('time_step') or 1
    match = re.search(r'(\d+)', str(value))
    return int(match.group(1)) if match else 1


def sample_reference_actors(pool, count, seed):
    if count < 1 or not pool:
        raise ValueError('A positive count and nonempty event interaction pool are required')
    rng = random.Random(seed)
    share_ratio = sum(content_type(r) == 'share' for r in pool) / len(pool)
    actors = []
    for draw in range(count):
        record = rng.choice(pool)
        actor = deepcopy(record)
        actor.setdefault('source_record_id', str(record.get('record_id') or record.get('id')
                                                or record.get('mid') or f'pool_{draw}'))
        actor.setdefault('source_mid', str(record.get('mid') or record.get('note_id') or record.get('id') or ''))
        actor.setdefault('source_t', record.get('t') or record.get('time_step') or record.get('time_step_index') or '')
        actor.setdefault('source_content_type', content_type(record))
        actor.setdefault('source_text', record_text(record))
        actor.setdefault('source_original_index', record.get('source_original_index', record.get('original_index', draw)))
        actor.update(source_reuse_round=0, source_is_reused=False,
                     source_sampling_mode='event_random_pool', source_random_draw_index=draw,
                     source_random_pool_size=len(pool), source_random_action_pool_size=len(pool),
                     source_random_share_ratio=share_ratio, source_random_seed=seed)
        actors.append(actor)
    return actors


def warmup_output(profile):
    content = '转发微博' if '媒体' in profile or '官方' in profile else '关注后续进展。'
    return dict(action='share' if content == '转发微博' else 'comment', content=content,
                raw_text=content, valid=False, trajectory_score=None, sft_log_likelihood=None,
                selector_mode='warmup_fallback', candidate_count=1, all_candidates=[])

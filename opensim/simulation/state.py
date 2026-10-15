"""Trained latent-state inference and recursive state summaries."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from opensim.state_transition.dataset import (
    TextFeatureEncoder, agent_numeric_features, clean_text, day_numeric_features,
    dist_from_dict, profile_text, text_hash_features,
)
from opensim.state_transition.model import FutureAwareLatentSTNet

LEAKY_AGENT_FIELDS = {'action', 'action_type', 'action_label', 'generated_action', 'is_comment', 'is_share', 'content_type', 'source_content_type', 'text', 'content', 'original_text', 'source_text', 'comments_count', 'shared_count', 'reposts_count', 'attitudes_count', 'liked_count', 'like_count', 'engagement_count', 'comment_level'}

def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate

def load_state_model(checkpoint_path: str | Path, device: torch.device, project_root: str | Path, *, base_model: str | None = None, cache_path: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root)
    checkpoint_path = _resolve_project_path(checkpoint_path, project_root)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    dims = checkpoint['dims']
    model = FutureAwareLatentSTNet(day_dim=int(dims['day_dim']), agent_dim=int(dims['agent_dim']), d_model=int(config.get('d_model', 192)), latent_dim=int(config.get('latent_dim', 64)), nhead=int(config.get('nhead', 6)), num_layers=int(config.get('num_layers', 2)), dropout=float(config.get('dropout', 0.1)), feature_dropout=float(config.get('feature_dropout', 0.0)), max_len=max(64, int(config.get('max_len', 64))), k_steps=int(config.get('k_steps', 3)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    encoder_path = base_model or str(_resolve_project_path(config.get('text_encoder_path', ''), project_root))
    text_encoder = TextFeatureEncoder(model_path=str(encoder_path), cache_db=cache_path, device=str(device), max_length=int(config.get('text_max_length', 64)), batch_size=int(config.get('text_batch_size', 16)))
    return {'kind': 'latent', 'model': model, 'text_encoder': text_encoder, 'config': config, 'dims': dims, 'checkpoint_path': str(checkpoint_path)}

def _day_feature_vector(bundle: dict[str, Any], rows: list[dict[str, Any]], day_idx: int) -> list[float]:
    row = rows[day_idx]
    inp = row.get('input') or {}
    prev_mu = dist_from_dict(rows[day_idx - 1].get('cumulative_state_distribution')) if day_idx > 0 else [0.0, 1.0, 0.0]
    event_text = clean_text(inp.get('event'))
    disturb_text = ' '.join((clean_text(value) for value in inp.get('disturb_names') or []))
    previous_mf = clean_text(inp.get('previous_mf_summary'))
    encoder = bundle['text_encoder']
    return prev_mu + encoder.encode(f'{event_text} {disturb_text}'.strip()) + encoder.encode(previous_mf) + day_numeric_features(rows, day_idx)

def _causal_agent_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep profile fields while removing current-response labels/outcomes."""
    causal = {key: value for key, value in record.items() if key not in LEAKY_AGENT_FIELDS}
    return causal

def _agent_feature_vector(record: dict[str, Any]) -> list[float]:
    causal_record = _causal_agent_record(record)
    features = text_hash_features(profile_text(causal_record)) + agent_numeric_features(causal_record)
    if len(features) >= 2:
        features[-2:] = [0.0, 0.0]
    return features

def build_policy_profile(record: dict[str, Any]) -> str:
    """Build the causal profile text used by policy training and inference."""
    causal_record = _causal_agent_record(record)
    return profile_text(causal_record) or '未知画像'

@torch.no_grad()
def predict_state(bundle: dict[str, Any], rows: list[dict[str, Any]], day_idx: int, records: list[dict[str, Any]], device: torch.device) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError('state model inference requires at least one agent record')
    day_vectors = [_day_feature_vector(bundle, rows, index) for index in range(day_idx + 1)]
    day_features = torch.tensor(day_vectors, dtype=torch.float32, device=device).unsqueeze(0)
    batch_size = len(records)
    agent_dim = int(bundle['dims']['agent_dim'])
    agent_features = torch.zeros((1, day_idx + 1, batch_size, agent_dim), dtype=torch.float32, device=device)
    agent_mask = torch.zeros((1, day_idx + 1, batch_size), dtype=torch.float32, device=device)
    current = torch.tensor([_agent_feature_vector(record) for record in records], dtype=torch.float32, device=device)
    agent_features[0, day_idx] = current
    agent_mask[0, day_idx] = 1.0
    day_mask = torch.ones((1, day_idx + 1), dtype=torch.float32, device=device)
    outputs = bundle['model'](day_features=day_features, agent_features=agent_features, agent_mask=agent_mask, day_mask=day_mask)
    return {'state_dist': outputs['state_dist'][0, day_idx].float().cpu().numpy(), 'state_latent': outputs['state_latent'][0, day_idx].float().cpu().numpy(), 'daily_pred': outputs['daily_pred'][0, day_idx].float().cpu().numpy(), 'future_daily_pred': outputs['future_daily_pred'][0, day_idx].float().cpu().numpy(), 'future_cumulative_pred': outputs['future_cumulative_pred'][0, day_idx].float().cpu().numpy(), 'future_delta_pred': outputs['future_delta_pred'][0, day_idx].float().cpu().numpy(), 'future_impact_pred': outputs['future_impact_pred'][0, day_idx].float().cpu().numpy()}

def latent_state_descriptor(state_dist: np.ndarray, future_delta: np.ndarray | None, future_impact: np.ndarray | None, state_latent: np.ndarray | None=None, latent_clusterer: Any | None=None) -> str:
    pos, neu, neg = np.asarray(state_dist, dtype=float).tolist()
    label, confidence = max((('积极', pos), ('中立', neu), ('消极', neg)), key=lambda item: item[1])
    if confidence >= 0.7:
        stance = f'强{label}'
    elif confidence >= 0.5:
        stance = f'偏{label}'
    else:
        stance = '混合观望'
    impact_text = '未知'
    if future_impact is not None and np.asarray(future_impact).size:
        impact = float(np.asarray(future_impact, dtype=float).mean())
        impact_text = '高' if impact >= 0.25 else '中' if impact >= 0.1 else '低'
    trend_text = '不明显'
    if future_delta is not None and np.asarray(future_delta).size:
        delta = np.asarray(future_delta, dtype=float).mean(axis=0)
        trend_idx = int(np.argmax(np.abs(delta)))
        if abs(float(delta[trend_idx])) >= 0.03:
            trend_name = ('积极', '中立', '消极')[trend_idx]
            trend_text = f"{trend_name}{('上升' if delta[trend_idx] > 0 else '下降')}"
    descriptor = f'{stance}；态度分布：积极{pos:.0%}，中立{neu:.0%}，消极{neg:.0%}；未来影响强度：{impact_text}；未来变化倾向：{trend_text}'
    if state_latent is not None and latent_clusterer is not None and np.asarray(state_latent).size:
        latent_code = int(latent_clusterer.predict(np.asarray(state_latent, dtype=np.float32).reshape(1, -1))[0])
        descriptor = f'潜在状态码：L{latent_code:02d}；{descriptor}'
    return descriptor

def normalize_mean_field(value: Any) -> str:
    if isinstance(value, list):
        return ' '.join((clean_text(item) for item in value if clean_text(item)))
    return clean_text(value)

def update_shadow_state(*, rows: list[dict[str, Any]], day_idx: int, daily_dist: list[float] | np.ndarray, cumulative_dist: list[float] | np.ndarray, mean_field_summary: str) -> None:
    daily = np.asarray(daily_dist, dtype=float)
    cumulative = np.asarray(cumulative_dist, dtype=float)
    rows[day_idx]['daily_state_distribution'] = {'pos': float(daily[0]), 'neu': float(daily[1]), 'neg': float(daily[2])}
    rows[day_idx]['cumulative_state_distribution'] = {'pos': float(cumulative[0]), 'neu': float(cumulative[1]), 'neg': float(cumulative[2])}
    rows[day_idx]['mf_summary'] = normalize_mean_field(mean_field_summary)
    if day_idx + 1 < len(rows):
        next_input = rows[day_idx + 1].setdefault('input', {})
        next_input['previous_mf_summary'] = normalize_mean_field(mean_field_summary)

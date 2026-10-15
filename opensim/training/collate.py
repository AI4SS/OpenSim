import torch


def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理 PolicyMFDataset 返回的字典列表。

    Args:
        batch: List[Dict], 每个 dict 包含:
            - input_ids: Tensor [seq_len]
            - attention_mask: Tensor [seq_len]
            - labels: Tensor [seq_len]
            - current_state: Tensor [3]
            - future_states: Tensor [K, 3]
            - question: str
            - response: str
            - file_name: str
            - idx: int

    Returns:
        Dict[str, Tensor/str]: batched data
    """
    def pad_1d(tensors, pad_value):
        max_len = max(tensor.shape[0] for tensor in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for row_idx, tensor in enumerate(tensors):
            out[row_idx, : tensor.shape[0]] = tensor
        return out

    # 动态 padding 到 batch 内最长序列，避免所有样本硬 pad 到 max_len。
    pad_token_id = batch[0].get('pad_token_id', 0)
    input_ids = pad_1d([item['input_ids'] for item in batch], pad_token_id)
    attention_mask = pad_1d([item['attention_mask'] for item in batch], 0)
    labels = pad_1d([item['labels'] for item in batch], -100)
    action_token_mask = pad_1d(
        [item.get('action_token_mask', torch.zeros_like(item['labels'], dtype=torch.bool)) for item in batch],
        False,
    )
    current_state = torch.stack([item['current_state'] for item in batch])
    future_states = torch.stack([item['future_states'] for item in batch])

    # 保持字符串和整数字段为列表
    questions = [item['question'] for item in batch]
    responses = [item['response'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    idxs = [item['idx'] for item in batch]
    action_types = [item.get('action_type', '') for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'action_token_mask': action_token_mask,
        'action_decision_index': torch.tensor(
            [item.get('action_decision_index', -1) for item in batch],
            dtype=torch.long,
        ),
        'action_class': torch.tensor(
            [item.get('action_class', 0) for item in batch],
            dtype=torch.long,
        ),
        'current_state': current_state,
        'future_states': future_states,
        'question': questions,
        'response': responses,
        'file_name': file_names,
        'idx': idxs,
        'action_type': action_types,
    }

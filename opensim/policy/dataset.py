import torch
from torch.utils.data import Dataset
import pandas as pd
import json

from opensim.policy.prompting import (
    POLICY_FORMAT_ACTION_CONTENT,
    POLICY_FORMATS,
    action_name_zh,
    build_policy_prompt_content,
    head_tail_truncate_prompt_ids,
    normalize_action,
    wrap_qwen_chat,
)


class PolicyMFDataset(Dataset):
    """Causal policy examples with response-only supervision and future masks.

    States use [positive, neutral, negative]. Prompts share the runtime template.
    """
    def __init__(
        self,
        data,
        tokenizer,
        max_length=1024,
        k_steps=10,
        policy_format=POLICY_FORMAT_ACTION_CONTENT,
    ):
        if policy_format not in POLICY_FORMATS:
            raise ValueError(f"Unsupported policy format: {policy_format}")
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_steps = k_steps
        self.policy_format = policy_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # =================================================================
        # 1. 字段提取
        # =================================================================
        topic = item.get('topic', '')

        # 优先使用 profile_text
        user_profile_str = item.get('profile_text', item.get('user_profile', '未知画像'))

        # 允许为空的字段
        hot_comment = item.get('hot_comment', None)
        related_cases_info = item.get('related_cases_info', None)
        root_post = item.get('root_post', None)
        latent_state_descriptor = item.get('latent_state_descriptor', None)

        # 舆论场数据 (兼容 字符串 或 列表)
        mean_field = item.get('mf_text', '')
        if isinstance(mean_field, list) and len(mean_field) > 0:
            mf_text = mean_field[-1]
        else:
            mf_text = mean_field

        # =================================================================
        # 2. 构建与仿真共用的因果 Prompt
        # =================================================================
        prompt_content = build_policy_prompt_content(
            topic=str(topic),
            root_post=str(root_post or ""),
            mean_field=str(mf_text or ""),
            profile=str(user_profile_str or "未知画像"),
            latent_descriptor=str(latent_state_descriptor or ""),
            hot_comment=str(hot_comment or ""),
            related_cases_info=str(related_cases_info or ""),
            policy_format=self.policy_format,
        )
        full_prompt_str = wrap_qwen_chat(prompt_content)

        # =================================================================
        # 3. 构造 Target，并显式标注自主动作 token
        # =================================================================
        eos_text = "<|im_end|>"
        action_token_offset = -1
        action_token_count = 0
        action = normalize_action(item.get("action_type", ""))
        prefix_ids = self.tokenizer(
            "动作：",
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        action_ids = self.tokenizer(
            action_name_zh(action),
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        content = str(item["real_comments"] or "").strip()
        if action == "share" and not content:
            content = "转发微博"
        suffix_ids = self.tokenizer(
            f"\n内容：{content}{eos_text}",
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        target_ids = prefix_ids + action_ids + suffix_ids
        action_token_offset = len(prefix_ids)
        action_token_count = len(action_ids)
        target_text = (
            f"动作：{action_name_zh(action)}\n内容：{content}{eos_text}"
        )

        # =================================================================
        # 4. Tokenize. Preserve the supervised response when truncating.
        # =================================================================
        prompt_ids = self.tokenizer(
            full_prompt_str,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length]
            prompt_ids = []
        else:
            prompt_budget = self.max_length - len(target_ids)
            prompt_ids = head_tail_truncate_prompt_ids(
                prompt_ids,
                prompt_budget,
            )

        input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # =================================================================
        # 5. 构造 Labels (Masking)
        # =================================================================
        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_ids):] = torch.tensor(target_ids, dtype=torch.long)
        action_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_decision_index = -1
        if action_token_offset >= 0 and action_token_offset < len(target_ids):
            action_start = len(prompt_ids) + action_token_offset
            action_end = min(
                action_start + action_token_count,
                len(input_ids),
            )
            action_token_mask[action_start:action_end] = True
            action_decision_index = action_start

        # =================================================================
        # 6. 状态数据 (MLP 部分 - 用于状态预测)
        # =================================================================
        # canonical order: [Pos, Neu, Neg].
        current_state = torch.tensor(item['user_state'], dtype=torch.float32)

        # future_states: [K, 3] -> 未来 K 步的真实分布
        future_states = torch.tensor(item['future_states_ground_truth'], dtype=torch.float32)
        future_mask = torch.tensor(item['future_mask'], dtype=torch.float32)
        st_latent = torch.tensor(item['st_latent'], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "action_token_mask": action_token_mask,
            "action_decision_index": action_decision_index,
            "action_class": (
                1 if normalize_action(item.get("action_type", "")) == "share" else 0
            ),
            "current_state": current_state,
            "future_states": future_states,
            "future_mask": future_mask,
            "st_latent": st_latent,
            # 保存原始文本供后续使用
            "question": full_prompt_str,
            "response": target_text,
            "file_name": item.get("file_name", "unknown"),
            "idx": idx,
            "event_id": item.get("event_id", ""),
            "day_index": int(item.get("day_index", -1)),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            "action_type": item.get("action_type", ""),
            "policy_format": self.policy_format,
        }


def load_and_process_csv_data(data_dir_or_file):
    """
    处理CSV数据，转换为PolicyMFDataset期望的格式。

    映射逻辑：
    - user_profile: profile_text
    - user_state: ST state_dist [pre_pos, pre_neu, pre_neg]
    - topic: topic
    - mf_text: batch_mf
    - real_comments: real_comments
    - future_states_ground_truth: 提取 dist_t0 到 dist_t10 的状态序列
    """
    import os
    import json

    if data_dir_or_file and os.path.exists(data_dir_or_file):
        if os.path.isdir(data_dir_or_file):
            # 目录模式，合并所有csv
            all_records = []
            for fname in sorted(os.listdir(data_dir_or_file)):
                if fname.endswith(".csv"):
                    fpath = os.path.join(data_dir_or_file, fname)
                    df = pd.read_csv(fpath)
                    processed_records = _process_csv_data(df)
                    all_records.extend(processed_records)
            print(f"Loaded {len(all_records)} samples from {data_dir_or_file}")
            return all_records
        elif data_dir_or_file.endswith(".csv"):
            df = pd.read_csv(data_dir_or_file, low_memory=False)
            processed_records = _process_csv_data(df)
            print(f"Loaded {len(processed_records)} samples from {data_dir_or_file}")
            return processed_records
        else:
            with open(data_dir_or_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        print("No data path provided or file not found.")
        return []


def _process_csv_data(df):
    """
    处理CSV数据，转换为PolicyModelDataset期望的格式。
    """
    processed_records = []

    for idx, row in df.iterrows():
        # 1. ST state_dist canonical order: [Pos, Neu, Neg].
        user_state = [
            float(row['pre_pos']),
            float(row['pre_neu']),
            float(row['pre_neg'])
        ]

        # 2. Strict future trajectory t+1...t+K in canonical order.
        future_states = []
        future_mask = []
        for k in range(10):
            pos_col = f'dist_t{k}_pos'
            neg_col = f'dist_t{k}_neg'
            neu_col = f'dist_t{k}_neu'

            if pos_col in row and pd.notna(row[pos_col]):
                future_states.append([
                    float(row[pos_col]),
                    float(row[neu_col]),
                    float(row[neg_col])
                ])
                valid_col = f'future_valid_t{k}'
                future_mask.append(
                    float(row[valid_col])
                    if valid_col in row and pd.notna(row[valid_col])
                    else 1.0
                )

        raw_latent = row.get('st_latent', row.get('_latent_vector', '[]'))
        if isinstance(raw_latent, str):
            st_latent = json.loads(raw_latent)
        elif isinstance(raw_latent, (list, tuple)):
            st_latent = list(raw_latent)
        else:
            st_latent = []

        def optional_text(name, default=""):
            if name not in row or pd.isna(row[name]):
                return default
            return str(row[name])

        # 3. 构建 Dataset 所需的记录
        record = {
            "user_profile": optional_text('profile_text', '未知画像'),
            "user_state": user_state,
            "topic": optional_text('topic'),
            "mf_text": optional_text('batch_mf'),
            "real_comments": optional_text('real_comments'),
            "future_states_ground_truth": future_states,
            "future_mask": future_mask,
            "st_latent": [float(value) for value in st_latent],
            "file_name": f"{row.get('uid', 'unknown')}.csv" if 'uid' in row else f"data_{idx}.csv",
            "idx": idx,
            "event_id": optional_text('event_id'),
            "day_index": int(row['day_index']) if 'day_index' in row and pd.notna(row['day_index']) else -1,
            "action_type": optional_text('action_type'),
            "root_post": optional_text('root_post'),
            "hot_comment": optional_text('hot_comment'),
            "related_cases_info": optional_text('related_cases_info'),
            "latent_state_descriptor": optional_text('latent_state_descriptor'),
        }

        processed_records.append(record)

    return processed_records

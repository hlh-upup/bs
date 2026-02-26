#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import json

# 确保可以导入项目根目录下的包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset import create_dataloaders_from_pkl

try:
    config = yaml.safe_load(open(os.path.join(ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8'))
except Exception as e:
    print(json.dumps({'error': f'load_config_failed: {e}'}))
    sys.exit(1)

try:
    dataset_path = os.path.join(ROOT, 'datasets', 'ac.pkl')
    train_loader, val_loader, test_loader = create_dataloaders_from_pkl(dataset_path, config)
except Exception as e:
    print(json.dumps({'error': f'create_dataloaders_failed: {e}'}))
    sys.exit(1)

try:
    batch = next(iter(train_loader))
    feats, labels, vids = batch
    out = {'features': {}, 'labels': {}, 'video_id_sample0': None}
    for k, v in feats.items():
        try:
            out['features'][k] = {'shape': tuple(v.shape), 'dtype': str(v.dtype)}
        except Exception as e:
            out['features'][k] = {'error': str(e)}
    for k, v in labels.items():
        try:
            out['labels'][k] = {'shape': tuple(v.shape), 'dtype': str(v.dtype), 'sample0': float(v[0].item())}
        except Exception as e:
            out['labels'][k] = {'error': str(e)}
    try:
        out['video_id_sample0'] = vids[0] if isinstance(vids, (list, tuple)) else vids
    except Exception:
        out['video_id_sample0'] = str(vids)
    print(json.dumps(out, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'error': f'batch_inspect_failed: {e}'}))
    sys.exit(1)

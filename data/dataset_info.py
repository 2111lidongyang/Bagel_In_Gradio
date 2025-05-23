# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '',
            'num_files': 0, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 0,
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '',
            'num_files': 0,
            'num_total_samples': 0,
            "parquet_info_path": '',
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '',
			'jsonl_path': '',
			'num_total_samples': 0
		},
    },
}

---
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: response
    dtype: string
  - name: category
    dtype: string
  - name: source
    dtype: string
  - name: judge_score
    struct:
    - name: prompt_is_harmful
      dtype: float64
    - name: response_is_refusal_or_dodge
      dtype: float64
    - name: response_contains_substantive_harm
      dtype: float64
    - name: jailbreak_label
      dtype: float64
    - name: rationale
      dtype: string
    - name: source
      dtype: string
  - name: label
    dtype: int64
  - name: activation_layer_4
    list: float64
  - name: activation_layer_10
    list: float64
  - name: activation_layer_15
    list: float64
  - name: activation_layer_20
    list: float64
  - name: activation_layer_25
    list: float64
  - name: activation_layer_30
    list: float64
  splits:
  - name: train
    num_bytes: 851310228
    num_examples: 4315
  - name: validation
    num_bytes: 121533058
    num_examples: 616
  - name: test
    num_bytes: 243456010
    num_examples: 1234
  download_size: 1214667407
  dataset_size: 1216299296
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

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
  - name: activation_layer_5
    list: float64
  - name: activation_layer_10
    list: float64
  - name: activation_layer_15
    list: float64
  - name: activation_layer_20
    list: float64
  - name: activation_layer_25
    list: float64
  splits:
  - name: train
    num_bytes: 399473169
    num_examples: 4292
  - name: validation
    num_bytes: 57064123
    num_examples: 613
  - name: test
    num_bytes: 114178962
    num_examples: 1227
  download_size: 568499883
  dataset_size: 570716254
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

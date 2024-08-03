# Utility tools for General Text Extraction

Utility tools for text extraction

## Install Requirement

```bash
pip install gensim
```

## Usage

This is a packages of functions for text extraction, which includes:

1. has_numbers: Function to detect if string contain any digit,
2. get_relation_numeric_unit: Function to split string of number into value and unit, also identify the comparative relation (more than, less than, approximately equal, equal)
3. get_similarity: Function to calculate string similarity between test string and reference list of strings, return (the most similar string from reference list, similarity score) which greater than given similarity threshold.

### get_relation_numeric_unit

### get_similarity

Most propotype algorithms can be found in `research/metric_mapper.py`.

It has been tested that by mapping both the input test samples and the target Data Field List into lowercase space, we can enhance the performance of the metric mapper model.

Hence, we suggest lower the data fields' tokens before mapping.

1. Build a upper case dictionary

```python
data_fields_upper_dict = build_upper_dict(data_fields_list)
```

This store the mapping relation between lowered data fields and their origin names, for example:

```python
{"total carbon emissions": "Total Carbon Emissions"}
```

2. Build target tokens:

```python
data_field_tokens_list = build_data_field_tokens_list(data_fields_list)
```

Note that `lower` is set to `True` in the API:

```
def build_data_field_tokens_list(data_fields_list, lower=True):
```

3. Build test words' tokens:

```python
src_words_tokens = [word.split() for word in src_words]
```

4. Build model

```python
def build_model(src_words_tokens, data_field_tokens_list):

    # Build the Model
    model = Word2Vec(sentences=src_words_tokens+data_field_tokens_list, vector_size=100,
                     window=5, min_count=1, workers=4)

    return model

model = build_model(src_words_tokens, data_field_tokens_list)
```

5. Matching

```python
for sample in src_words_tokens:
    df_this_sample = create_match_table(model, sample, data_field_tokens_list)

    target_by_distance, min_distance = retrieve_target(df_this_sample, by="distance")

    target_by_similarity, max_similarity = retrieve_target(df_this_sample, by="similarity")
```

As an option, you can convert the matched target back to original lower and upper cases:

```python
data_fields_upper_dict(target_by_distance)
data_fields_upper_dict(target_by_similarity)
```

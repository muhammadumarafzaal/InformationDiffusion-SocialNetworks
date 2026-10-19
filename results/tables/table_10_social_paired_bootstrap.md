**Table 10: paired bootstrap of per-user MSE and NDCG@10 differences (test fold)**

| metric   | comparison         |   mean_difference |   ci_low |   ci_high | significant   | favours    |   users |
|:---------|:-------------------|------------------:|---------:|----------:|:--------------|:-----------|--------:|
| mse      | trust - similarity |           -0.0036 |  -0.0085 |    0.0008 | False         | trust      |   10706 |
| mse      | trust - random     |           -0.0054 |  -0.0100 |   -0.0010 | True          | trust      |   10706 |
| ndcg@10  | trust - similarity |           -0.0099 |  -0.0114 |   -0.0083 | True          | similarity |   10706 |
| ndcg@10  | trust - random     |            0.0018 |   0.0008 |    0.0030 | True          | trust      |   10706 |

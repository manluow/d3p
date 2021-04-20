# D3P: Data-driven Demand Prediction for Fast Expanding Electric Vehicle Sharing Systems

Our paper can be found [here](http://wrap.warwick.ac.uk/135568/1/WRAP-D3P-data-driven-demand-prediction-fast-expanding-electric-vehicle-sharing-systems-Wen-2020.pdf).

## Prerequisites


```
- python=3.6   
- numpy
- pandas  
- tensorflow-gpu=1.8.0  
- scikit-learn  
```

## Data Preparation

We've provided sample data in `./data`, which can be extracted by:

```python data_preparing.py```

You could also plug in your own data in the d3p framework.

## Demand Prediction

### Predicting expected demand using `d3p-exp`:

Generate data:

```python seq2avg_data_generation.py```

Training:

```python seq2avg_train.py```

Testing:

```python seq2avg_predict.py```

### Predicting instant demand using `d3p-seq`:

Generate data:
```python seq2seq_data_generation.py```

Training

```python seq2seq_train.py```

Testing

```python seq2seq_predict.py```



## Citations
```
@article{Luo:IMWUT:2020, 
author = {Luo, Man and Du, Bowen and Klemmer, Konstantin and Zhu, Hongming and Ferhatosmanoglu, Hakan and Wen, Hongkai}, 
title = {D3P: Data-Driven Demand Prediction for Fast Expanding Electric Vehicle Sharing Systems}, 
year = {2020}, 
issue_date = {March 2020}, 
publisher = {Association for Computing Machinery}, 
address = {New York, NY, USA}, 
volume = {4}, 
number = {1}, 
url = {https://doi.org/10.1145/3381005},
```


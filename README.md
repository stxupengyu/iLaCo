

Official code for our paper.  

## Requirements: 
* python==3.7.10
* numpy==1.21.2
* scipy==1.7.2
* scikit-learn==0.22
* torch==1.10.0
* gensim==4.1.2
* nltk==3.6.5
* tqdm==4.62.3 


## Datasets
All datasets are derived from [LSAN](https://aclanthology.org/D19-1044/) and [LDGN](https://aclanthology.org/2021.acl-long.298/). 


## Experiments

### Data Path
Please confirm the corresponding configuration file. Make sure the data path parameters (data_dir, dataset and etc.) are right in:   
```bash
main.py
```

### train and eval
```bash
bash script/run_ilaco.sh <dataset> <noise_rate> <gpu_id> 
```

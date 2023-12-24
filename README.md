
# run

## 42 servers
/data/pengyu/anaconda3/envs/pt19/bin/python /home/pengyu/code/nmll/ilaco/main.py  --gpu 7 --data_name data_rho0.6.txt  
/data/pengyu/anaconda3/envs/pt19/bin/python /home/pengyu/code/nmll/ilaco/main.py  --gpu 6 --data_name data_rho0.4.txt
/data/pengyu/anaconda3/envs/pt19/bin/python /home/pengyu/code/nmll/ilaco/main.py  --gpu 5 --data_name data_rho0.2.txt
/data/pengyu/anaconda3/envs/pt19/bin/python /home/pengyu/code/nmll/ilaco/main.py  --gpu 4 --data_name data_rho0.2.txt  --pretrained True



## 18 servers
/home/pengyu/anaconda3/envs/pt/bin/python /home/pengyu/code/nmll/ilaco/main.py  --gpu 7 --mode rcn --rate10 0.0 --pretrained False


--data_name data_rho0.6.txt

# bash
bash /home/pengyu/code/nmll/tabasco/run_tuning.sh rcn 0
bash /home/pengyu/code/nmll/tabasco/run_tuning.sh 


# scp to another server
scp -r /data/pengyu/NMLL/physics pengyu@10.126.62.42:/data/pengyu/NMLL/
scp -r /home/pengyu/code/nmll/led/ pengyu@10.126.62.42:/home/pengyu/code/nmll/
scp -r /home/pengyu/code/nmll/ pengyu@10.126.62.42:/home/pengyu/code/
scp -r /home/pengyu/code/nmll/ pengyu@10.126.56.18:/home/pengyu/code/




---

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

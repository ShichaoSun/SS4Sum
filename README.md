# Improving Sentence Similarity Estimation for Unsupervised Extractive Summarization
This repo contains the code, data and trained models for our ICASSP 2023 paper ''Improving Sentence Similarity Estimation for Unsupervised Extractive Summarization''

## Requirements
- Python 3.7
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)

Install dependencies via:
```
conda create -n ss4sum python=3.8
conda activate ss4sum
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# For rouge-1.5.5.pl
sudo apt-get update
sudo apt-get install expat
sudo apt-get install libexpat-dev -y

sudo cpan install XML::Parser
sudo cpan install XML::Parser::PerlSAX
sudo cpan install XML::DOM

git clone https://github.com/summanlp/evaluation
pyrouge_set_rouge_path yourPath/evaluation/ROUGE-RELEASE-1.5.5
```
## Download the Datasets and Best Checkpoint

Download processed datasets and checkpoint from https://drive.google.com/drive/folders/17wxORu-xmLPzGKVzecpikXaZeRkrjjmo?usp=sharing

The original datasets can be found at https://github.com/mswellhao/PacSum.

## Train and Test
You may specify the hyper-parameters in exp/****.sh. 
We also provide the specific settings (train on CNNDM and NYT; test on CNNDM).

- Train: 
```
bash exp/train.sh
```

- Test on CNNDM:
```
bash exp/test.sh
```

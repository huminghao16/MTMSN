# A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning

This repo contains the code of the following paper:

<i> [A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning](https://arxiv.org/abs/1908.05514). Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li. EMNLP 2019.</i>

In this paper, we propose the Multi-Type Multi-Span Network (MTMSN) for reading comprehension that requires discrete reasoning.

The network contains: 
- a multi-type answer predictor that supports the prediction of various answer types (e.g., date, number, and span); 
- a multi-span extractor that dynamically produces one or multiple text strings; 
- an arithmetic expression reranking mechanism that re-ranks candidate expressions to further confirm the prediction.  


## Pre-trained Models
To reproduce our results, we release the following pre-trained models:
- [MTMSN_LARGE](https://drive.google.com/open?id=1cqvaBJIG9iPIOxp5VzzUTzGknaj9Vmbo)

## Requirements
- Python 3.6
- [Pytorch 1.1](https://pytorch.org/)
- [Allennlp 0.8.4](https://allennlp.org/)

Download the uncased [BERT-Base](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory. 

## Train
Make sure `drop_dataset_train.json` and `drop_dataset_dev.json` are placed in `DATA_DIR`.

Then set up the environment:
```bash
export DATA_DIR=data/drop
export BERT_DIR=bert-base-uncased
```

Run the following command to train the base model:
```shell
python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA_DIR/drop_dataset_train.json \
  --predict_file $DATA_DIR/drop_dataset_dev.json \
  --train_batch_size 12 \
  --predict_batch_size 24 \
  --num_train_epochs 10.0 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --span_extraction \
  --addition_subtraction \
  --counting \
  --negation \
  --gradient_accumulation_steps 2 \
  --output_dir out/mtmsn_base
```
The above model was trained on a single GPU with 16GB memory. Once the training is done, you can check out the dev result in `out/mtmsn_base/performance.txt`.

To train the large model, make sure there are 4 GPUs with 16GB memory per card and run the following command:
```shell
python -m bert.run_mtmsn \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA_DIR/drop_dataset_train.json \
  --predict_file $DATA_DIR/drop_dataset_dev.json \
  --train_batch_size 24 \
  --predict_batch_size 48 \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --span_extraction \
  --addition_subtraction \
  --counting \
  --negation \
  --gradient_accumulation_steps 2 \
  --optimize_on_cpu \
  --output_dir out/mtmsn_large
```

## Acknowledgements
Our implementation is based on the [naqanet](https://github.com/allenai/allennlp/blob/master/allennlp/models/reading_comprehension/naqanet.py) model.

If you find the paper or this repository helpful in your work, please use the following citation:
```
@inproceedings{hu2019multi,
  title={A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning},
  author={Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng},
  booktitle={Proceedings of EMNLP},
  year={2019}
}
```

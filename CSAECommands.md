# Commands

### Project Set Up

```
conda create -n py37 -y python=3.7 pip
conda activate py37
pip install setuptools wheel numpy pandas
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge spacy=2.1.7
pip install pytorch-transformers==1.1.0
```

### Run
```
python run_ae.py --bert_model roberta-base --data_dir /home/zehong/CSAE-code/ae/laptop --output_dir /home/zehong/CSAE-code/src/outputs --max_seq_length 128 --do_train --do_valid –do_eval --train_batch_size 32 --learning_rate 3e-5 --pos_embedding_size 32 --num_train_epochs 20
```

### Evaluate
```
python eval/evaluate.py --pred /home/zehong/CSAE-code/src/outputs/predictions.json --target /home/zehong/CSAE-code/data/laptop/laptops--test.gold.xml
```
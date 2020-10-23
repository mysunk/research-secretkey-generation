### Will be updated:
* citation link
* summary
* dataset description
* result  
* __PLZ REBASE COMMIT LOG!!!__

Deep Learning based end-to-end Physical Layer Secret Key Generation using Wireless Channel State Information
=======================================
This repository is the official implementation of Deep Learning based end-to-end Physical Layer Secret Key Generation using Wireless Channel State Information.  

![overview_keygen](https://user-images.githubusercontent.com/48520885/101493782-d920e500-39a9-11eb-8d62-2330d9dbbf87.png)

Requirements
=======================================
To install requirements:
```setup
pip install -r requirements.txt
```

Datasets
==================

Training and Evaluation
==================
There is two options to training model and evaluate the result.
```train and eval
python main.py --result_save_dir <path-to-dir> --EPOCHS 1000 --N_POPULATION 100 --N_BEST 10 --h1 64 --h2 64 --h3 64 --early_stopping 50 --POWER_RATIO 0.5 --CONST 0.8
python evaluate.py --result_save_dir <path-to-dir> --reference 1 --POWER_RATIO 0.5 --CONST 0.8
```

or just
```train
sh run.sh
```
Pre-trained models
==================
You can use the pre-trained model without re-training from the beginning.  
It can be found in 'results' dir and just specify the result_save_dir in evaluate.py argument to it.
```eval with pre-trained model
python evaluate.py --result_save_dir results --reference 1 --POWER_RATIO 0.5 --CONST 0.8
```

Results
==================

# Inverse Propensity Scoring Matrix Factorization with Pytorch-lightning

In this repo, I implemented Inverse Propensity Matrix Factorization for Movielens dataset for explicit ratings under MSE loss. The objective is to predict the value of the rating for a given (user,item) ID. 



IPS methods in Recommender Systems are gaining more and more attention. In fact, these method help unbias the performance estimation of the algorithm by weighting it by the inverse propensity of each rating

Furthermore, as propensity is an unseen variable, we provide two methods to estimate its value: ***<u>Popularity based propensity</u>*** and ***<u>Poisson based propensity</u>*** These models are discussed in the following paper: http://www.its.caltech.edu/~fehardt/UAI2016WS/papers/Liang.pdf (Causal Inference for Recommendation)



## Requirements and installation

Clone this repo using
```bash
git clone https://github.com/samikhenissi/IPS_MF.git
```


We used pytorch 1.5.1 and pytorch-lightning==0.8.5

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements or refer to official website for [pytorch](https://pytorch.org/) and [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning).

```bash
pandas
numpy
torch
pytorch-lightning
hpfrec
```

You can also use  

```bash
pip install -r requirements.txt
```



## Usage

#### For training IPS_MF

The main training script is in train.py. You will need a training data in a pandas dataframe that has the following columns:  ['uid', 'mid', 'rating', 'timestamp']

You can try the implementation on Movielens-100K or Movielens-1m

For example, to run the training script using MF_IPS on the Movielens-100 data you can use:

```bash
train.py --model IPS_MF --data movielens100
```

pytorch-lightning allows scalable training on multi gpus. For more information refer to: [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html) 



#### To use Propensity models:

To use the implemented Propensity model you can load the Propensity module through:

```python
from Propensity import *
```

Then create a Propensity instance

```python
propensity = Propensity(propensity_model = poisson)
```

To train the propensity on a given dataframe (the columns of the dataframe need to refer to: UserId ,ItemId,rating in the same order)

```python
propensity.fit(data)
```

To predict the propensity on a new dataframe using the fitted model

```python
propensity.predict(newdata)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

To contact me with any concern you can also email me at sami.khenissi@louisville.edu
## License
[MIT](https://choosealicense.com/licenses/mit/)

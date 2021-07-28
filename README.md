# Let's agree to disagree: Consensus Entropy Active Learning for Personalized Music Emotion Recognition

Implementation of active learning for our paper presented at the 22nd International Society for Music Information Retrieval Conference (ISMIR).
-- Juan Sebastián Gómez-Cañón, Estefanía Cano, Yi-Hsuan Yang, Perfecto Herrera, and Emilia Gómez

## Abstract
Previous research in music emotion recognition (MER) has tackled the inherent problem of subjectivity through the use of personalized models -- models which predict the emotions that a particular user would perceive from music. Personalized models are trained in a supervised manner, and are tested exclusively with the annotations provided by a specific user. While past research has focused on model adaptation or reducing the amount of annotations required from a given user, we propose a methodology based on uncertainty sampling and query-by-committee, adopting prior knowledge from the agreement of human annotations as an oracle for active learning (AL). We assume that our disagreements define our personal opinions and should be considered for personalization. We use the DEAM dataset, the current benchmark dataset for MER, to pre-train our models. 
We then use the AMG1608 dataset, the largest MER dataset containing multiple annotations per musical excerpt, to re-train diverse machine learning models using AL and evaluate personalization. Our results suggest that our methodology can be beneficial to produce personalized classification models that exhibit different results depending on the algorithms' complexity. 

## Installation
Clone this repository:
```
git clone https://github.com/juansgomez87/consensus-entropy
cd consensus-entropy
```

Create a virtual environment using `venv` and install requirements:
```
python3 -m venv cons-env
source cons-env/bin/activate
pip3 install -r requirements.txt
```
Since we made a minor change to the [XGBoost library](https://xgboost.readthedocs.io/en/latest/), run the following command to allow for re-training XGB models:
```
cp xgboost/sklearn.py ./cons-env/lib/python3.8/site-packages/xgboost/
```
We also use [short-chunk CNN](https://github.com/minzwon/sota-music-tagging-models/) proposed by Won et al. Thank you to the authors!

*Note:* Since 

## Data

The DEAM dataset is available in [this link](https://cvml.unige.ch/databases/DEAM/). 



## Usage


## Publication
```
@InProceedings{GomezCanon2021ISMIR,
    author = {Juan Sebastián Gómez-Cañón and Estefanía Cano and Yi-Hsuan Yang and Perfecto Herrera and Emilia Gómez},
    title = {Let's agree to disagree: Consensus Entropy Active Learning for Personalized Music Emotion Recognition},
    booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR)},
    year = {2021},
    location = {Online},
    pages = {},
}
```

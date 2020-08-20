# Cat Segmentation

Cat segmentation experiment using Tensorflow2 TF2

## Datasets
- [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

    `curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz`

    `curl -O  http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz`

## Usage

- `pip install -r requirements.txt` ( TODO )

- Run `check_data.ipynb` -> `train_model.ipynb` -> `test_model.ipynb`

    - `check_data.ipynb` for preparing training data

    - `train_model.ipynb` for training segmentation model

    - `test_model.ipynb` for testing model on validation data or inference own cat pictures

## Current Result

**Accuracy and Loss**
![](https://i.imgur.com/8F0VDUS.png)

**Validation Image**
![](https://i.imgur.com/WfQS7DH.png)
![](https://i.imgur.com/FgYEmSR.png)

**Test Image**
![](https://i.imgur.com/DQuT9zV.png)

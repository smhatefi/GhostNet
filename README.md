# GhostNet
GhostNet implementation from scratch for classifying images from the Oxford-IIIT Pet Dataset

## Project Structure
- `ghostnet.py`: Contains the model.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the model.
- `ghostnet_pet_model.pth`: Pre-trained weights provided for your convenience.

## Usage

### 1. Training the Model
To train the model, run:
```
python train.py
```

### 2. Evaluating the Model
First specify the location of your desired image in `img_path` variable in `evaluate.py`

Then run the `evaluate.py` script to make predictions using the trained model:
```
python evaluate.py
```


## Dataset
The dataset used for training is the Oxford-IIIT Pet Dataset.

![Oxford-IIIT Pet Dataset Statistics](https://www.robots.ox.ac.uk/~vgg/data/pets/breed_count.jpg)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The GhostNet model architecture is inspired by the original [GhostNet paper](https://arxiv.org/abs/1911.11907).
- The code is based on the official [GhostNet repo](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch).

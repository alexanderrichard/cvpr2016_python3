# NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
Code for the paper NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning

### Prepraration:

* download the data from https://uni-bonn.sciebo.de/s/vVexqxzKFc6lYJx
* extract it so that you have the `data` folder in the same directory as `train.py`
* create a  `results` directory in the same directory where you also find `train.py`: `mkdir results`

Requirements: Python3.x with the libraries numpy and pytorch (version 0.4.1)

### Data:
The data folder contains 3 folders and 3 files:
- /features:
- /groundTruth: contains a file for each video where each line present a label assigned to a video-frame respectively
- /transcript: contains a file for each video, where an ordered sequence of classes that occurr in the video is given
- mapping.txt: contains a list of number assigned to each  action-label 
git - split1.test: list of videos used for training
- split1.train: list of videos used for testing
### Training:


Run `python3 train.py`

### Inference:

Run `python3 infernece.py`
Note: adjust the variable `n_threads` in `inference.py` to your needs.

### Evaluation:

In the inference step, recognition files are written to the `results` directory. The frame-level ground truth is available in `data/groundTruth`. Run `python eval.py --recog_dir=results --ground_truth_dir=data/groundTruth` to evaluate the frame accuracy of the trained model.

### Remarks:

We provide a python/pytorch implementation for easy usage. In the paper, we used a faster in-house C++ implementation, so results can be slightly different. Running the provided setup on split1 of Breakfast should lead to roughly 42% frame accuracy.

If you use the code, please cite

    A. Richard, H. Kuehne, A. Iqbal, J. Gall:
    NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
    in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018

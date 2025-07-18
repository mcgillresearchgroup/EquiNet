"""Loads a trained chemprop model checkpoint for vle or vp prediction and save the predicted function parameters
 for the molecules in a dataset. Uses the same command line arguments as predict."""

from chemprop.train import chemprop_parameters

if __name__ == '__main__':
    chemprop_parameters()

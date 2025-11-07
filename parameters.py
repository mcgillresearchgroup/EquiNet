"""Loads a trained equinet model checkpoint for vle or vp prediction and save the predicted function parameters
 for the molecules in a dataset. Uses the same command line arguments as predict."""

from equinet.train import equinet_parameters

if __name__ == '__main__':
    equinet_parameters()

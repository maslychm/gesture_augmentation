# Effective 2D Stroke-based Gesture Augmentation for RNNs

Implementation of the augmentation methods described in the paper "Effective 2D Stroke-based Gesture Augmentation for RNNs" by Mykola Maslych, Mostafa Aldilati, Dr. Eugene M. Taranta II, and Dr. Joseph J. LaViola Jr. (2023) <https://doi.org/10.1145/3544548.3581358>.

RNNs show great performance on time-series tasks, but specifically for custom gestures, the data provided by a user is not enough to train an accurate model. We evaluate a number of existing augmentation methods and chain them into series of transformations that maximize accuracy.  

## Environment Setup

Windows:

    conda create --name py10 python=3.10
    conda activate py10
    conda install numpy
    conda install scipy
    conda install matplotlib

Mac:

    conda create --name py10 python=3.10
    conda activate py10
    conda install numpy
    conda install pytorch -c pytorch
    conda install matplotlib
    conda install scipy
    pip install pytorch-lightning

## Citing

If you use this code for your research, please cite our paper:

    @inproceedings{maslych2023gesture_augmentation,
        author = {Maslych, Mykola and Aldilati, Mostafa and Taranta II, Eugene M. and LaViola Jr., Joseph J.},
        title = {Effective 2D Stroke-based Gesture Augmentation for RNNs},
        booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
        series = {CHI '23},
        year = {2023},
        location = {Hamburg, Germany},
        pages = {TODO},
        doi = {10.1145/3544548.3581358},
        publisher = {ACM},
    }

## Contributions and Bug Reports

Contributions are welcome. Please submit your contributions as a pull request and we will incorporate them. If you find any bugs, please report them as an issue.

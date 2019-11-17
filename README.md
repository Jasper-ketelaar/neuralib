# NeuraLib
Library for creating multi-layered perceptrons in Java. This project was started to create a Generative Adversial Network using multi-layer
perceptrons as the classifier and generator.

This GAN is being built to trick the cheat detection system in a game called Old School Runescape by generating human-like mouse data.

The process will be:
1) Building a classifier that can classify game data to be human or inhuman
2) Building a generator that will attempt to generate game data and feed it to the classifier
3) Improving both the classifier and the generator iteratively such that the generator can generate data to which the classifier does not know what answer to give.



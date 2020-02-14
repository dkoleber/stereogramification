## Stereogramification

This project provides a pipeline to create a stereogram from any image.

### Pipeline

A fully convolutional neural network (trained on NYU-Depth V2) produces a depth map which is fed into the stereogram algorithm described in [Displaying 3D Images: Algorithms for Single-Image Random-Dot Stereograms](https://www.cs.waikato.ac.nz/~ihw/papers/94-HWT-SI-IHW-SIRDS-paper.pdf).


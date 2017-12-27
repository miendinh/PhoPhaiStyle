## Pho Phai Style

* Notations:
  - C : Content image
  - S : Style image
  - G : Generated image
  - $\alpha$: hyperparameter weighting the importance of the content cost
  - $\beta$: hyperparameter weighting the importance of the style cost
  - H : Height of an image
  - W : Weight of an image
  - C : Number of channel ( 3 - RGB )
  - $\lambda^{[l]}$ are given in `STYLE_LAYERS`.
* The content cost function: $J_{content}(C,G)$
* The style cost function:  $J_{style}(S,G)$
* Integrate above cost functions:  $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$

$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum_{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1}$$

$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij} - G^{(G)}_{ij})^2\tag{2}$$


$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

* VGG19 Model Summary

| #Layer   |  Name   |      Weight  Dimension  |
| -------- |:--------|:---------------|
|  0     |      conv1_1| (3, 3, 3, 64)   |
|  1     |      relu ||
|  2     |      conv1_2| (3, 3, 64, 64)   |
|  3     |      relu ||
|  4     |      maxpool   ||
|  5     |      conv2_1| (3, 3, 64, 128)
|  6     |      relu   ||
|  7     |      conv2_2| (3, 3, 128, 128)
|  8     |      relu   ||
|  9     |      maxpool |
|  10     |      conv3_1| (3, 3, 128, 256)   |
|  11     |      relu ||
|  12     |      conv3_2| (3, 3, 256, 256)   |
|  13     |      relu ||
|  14     |      conv3_3| (3, 3, 256, 256)   |
|  15     |      relu ||
|  16     |      conv3_4| (3, 3, 256, 256)   |
|  17     |      relu ||
|  18     |      maxpool ||
|  19     |      conv4_1| (3, 3, 256, 512)
|  20     |      relu   ||
|  21     |      conv4_2| (3, 3, 512, 512) |
|  22     |      relu   ||
|  23     |      conv4_3| (3, 3, 512, 512) |
|  24     |      relu   ||
|  25     |      conv4_4| (3, 3, 512, 512) |
|  26     |      relu   ||
|  27     |      maxpool ||
|  28     |      conv5_1| (3, 3, 512, 512)   |
|  29     |      relu   ||
|  30     |      conv5_2| (3, 3, 512, 512)   |
|  31     |      relu   ||
|  32     |      conv5_3| (3, 3, 512, 512)   |
|  33     |      relu   ||
|  34     |      conv5_4| (3, 3, 512, 512)   |
|  35     |      relu   ||
|  36     |      maxpool ||
|  37     |      fullyconnected| (7, 7, 512, 4096)   |
|  38     |      relu ||
|  39     |      fullyconnected| (1, 1, 4096, 4096)   |
|  40     |      relu ||
|  41     |      fullyconnected| (1, 1, 4096, 1000)   |
|  42     |      softmax   ||   

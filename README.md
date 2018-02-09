## Pho Phai Style

#### 1. Notations
  - C : Content image
  - S : Style image
  - G : Generated image
  - ![](https://latex.codecogs.com/gif.latex?\inline&space;\alpha): hyperparameter weighting the importance of the content cost
  - ![$\beta$](https://latex.codecogs.com/gif.latex?\inline&space;\beta): hyperparameter weighting the importance of the style cost
  - H : Height of an image
  - W : Weight of an image
  - C : Number of channel ( 3 - RGB )
  - ![$\lambda^{[l]}$](https://latex.codecogs.com/gif.latex?\inline&space;$\lambda^{[l]}$) are given in style layers.

#### 2. The content cost function

 ![$J_{content}(C,G)$](https://latex.codecogs.com/gif.latex?\inline&space;$J_{content}(C,G)$)

#### 3. The style cost function

![$J_{style}(S,G)$](https://latex.codecogs.com/gif.latex?\inline&space;$J_{style}(S,G)$)

#### 4. Integrate above cost functions

  ![$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$](https://latex.codecogs.com/gif.latex?\inline&space;$J(G)&space;=&space;\alpha&space;J_{content}(C,G)&space;&plus;&space;\beta&space;J_{style}(S,G))

![$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum_{ \text{all entries}} (a^{(C)} - a^{(G)})^2 \tag{1}$$](https://latex.codecogs.com/gif.latex?J_{content}(C,G)&space;=&space;\frac{1}{4&space;\times&space;n_H&space;\times&space;n_W&space;\times&space;n_C}\sum_{&space;\text{all&space;entries}}&space;(a^{(C)}&space;-&space;a^{(G)})^2)

![$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij} - G^{(G)}_{ij})^2\tag{2}$$](https://latex.codecogs.com/gif.latex?J_{style}^{[l]}(S,G)&space;=&space;\frac{1}{4&space;\times&space;{n_C}^2&space;\times&space;(n_H&space;\times&space;n_W)^2}&space;\sum&space;_{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij}&space;-&space;G^{(G)}_{ij})^2)


![$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$](https://latex.codecogs.com/gif.latex?J(G)&space;=&space;\alpha&space;J_{content}(C,G)&space;&plus;&space;\beta&space;J_{style}(S,G))


#### 5. VGG19 Model Summary [link](VGG19.md)

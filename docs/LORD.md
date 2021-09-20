## LORD (Latent Optimization for Representation Disentanglement)



a model to disentangle the hidden factors of variations

*disentangling class and content representations assumes that intra-class variation is significantly lower than inter-class variation.*

**lord do not give any explicit constraint to the latent, so it can not generate new sample, but the disentanglement is outperform any other method.**

*what is inductive bias?*



a non-adversarial method to class and content disentanglement

there are two stage for this method:

##### the first stage:

use latent optimization for class supervision( not the amortized inference)

![LORD](./pic/LORD.png)

![LORD_0](./pic/LORD_0.png)

##### second stage :

use a forward network to learn the image latent code mapping



![LORD_1](./pic/LORD_1.png)


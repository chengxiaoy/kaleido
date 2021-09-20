## Soft-IntroVAE



**Objective**

the original IntroVAE loss function rely on a particular hinge-loss formulation that is very hard to stabilize in practice.

the objective of this paper

> replace the hinge-loss term with a smooth exponential loss on generated samples and understanding the IntroVAE model



**the paper's discovery:**

1. In contrast to the original IntroVAE, the S-IntroVAE encoder converges to the true posterior!

*who to define the true posterior? mock data?*

2. the soft-introVAE encoder converges to a generative distribution that minimizes a sum of KL divergence from the data distribution and a entropy term 

*not equivalent, what is the entropy term mean?*

3. S-IntroVAE effectively assigns a lower likelihood to out-of-distribution.

*a capacity to detect the outlier in samples?*



###### use image translation to demonstrate the advantage of model's inference.

the current SOTA in image translation is LORD, which used the class-supervised.

soft-IntroVAE could achieve the SOTA unsupervised image translation result closed to LORD.










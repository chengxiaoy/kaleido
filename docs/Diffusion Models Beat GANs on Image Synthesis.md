## Diffusion Models Beat GANs on Image Synthesis

*there are two contributions from this paper*

1. show diffusion can achieve image sample quality superior to the current state-of-the-art generative model by finding a better architecture through a series of ablations.
2. for conditional image synthesis,  improved sample quality with classifier guidance.



#### ***Sample Quality:***

*the methods in this paper to improve the sample quality*:

1. Learn from Imporved DDPMs paper, the log-likelihood is improved by predicting the variance $\sum_{\theta}(x_{t},t)$

so the objective could be changed to :
$$
L_{hybrid} = L_{simple}+ \lambda L_{vlb}
$$

2. Use DDIM sample approach when using fewer than 50 sampling steps.

3. Optimize the model architecture



 the most important part of this paper:

#### ***Classifier Guidance**:*

GAN for conditional image synthesis make heavy use of class labels.

**exploiting a classifier $p(y|x)$ to improve a diffusion generator.**

we can train a classifier $p_{\phi}(y|x_{t},t)$ on noisy images $x_{t}$, and then use gradients $\nabla_{x_t}\log p_{phi}(y|x_{t},t)$ to guide the diffusion sampling process towards an arbitrary class label $y$

![image-20221023102639129](pic\Conditional_Diffusion_Model.png)

for DDIMs, we have

![image-20221023103048156](pic\Conditional_DDIM.png)

 to apply classifier guidance to large scale generative task, classification models are trained on ImageNet;

guidance improves precision at the cost of recall, **thus introducing a trade-off in sample diversity versus fidelity.**

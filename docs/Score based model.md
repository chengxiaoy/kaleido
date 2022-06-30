## Score based generative models
> https://yang-song.github.io/blog/2021/score/

there are several advantages over existing model
- GAN-level sample quality without adversarial training
- flexible model architecture 
- exact log-likelihood computation

Existing generative modeling technology could be grouped into two categories based on how they represent probability 
distributions.
1. likelihood-based models, which directly learn the distribution's PDF 
typical likelihood-based models include autoregressive models, normalizing flow models, energy-based models(EBMs) and VAEs
   
2. implicit generative models, where the probability distribution is implicitly represented by a model of its sampling process


score-based models have connection to normalization flow models, therefore allowing exact likelihood computation and representation learning





likelihood-based models is to directly model the PDF, let $f_{\theta}(x)\in R$ be a real-valued function parameterized by a learned parameter $\theta$ :
$$
p_{\theta}(x) = \frac{e^{-f_{\theta}(x)}}{Z_{\theta}}
$$


where $Z_{\theta}$ is the normalizing constant to make sure $\int p_{\theta}(x)\,dx=1$



while the $Z_{\theta}$ is always intractable 

***To understand below point of view***

> Thus to make maximum likelihood training feasible, likelihood-based models must either restrict their model architectures (e.g., causal convolutions in autoregressive models, invertible networks in normalizing flow models) to make $Z_{\theta}$ tractable, or approximate the normalizing constant (e.g., variational inference in VAEs, or MCMC sampling used in contrastive divergence) which may be computationally expensive.



To sidestep the intractable normalizing constants, we can model the score function 

the **score function** of a distribution $p(x)$ is defined as 
$$
\nabla_{x}\log p(x)
$$
a model for the score function is called a score-based model, we denote as $s_{\theta}(x)$



we train score-based models by minimized the **Fisher divergence** between the model and data distribution
$$
E_{p(x)}[||\nabla_{x}\log p(x)-s_{\theta}(x)||^2]
$$
we can not directly compute the divergence as we do not know the real data score $\nabla_{x}\log p(x)$



there exists a family of methods called **score matching**  that minimize the Fisher divergence without knowledge of the real data score 



***Langevin dynamics*** 

langevin dynamic provides an MCMC procedure to sample from a distribution $p(x)$ using only its score function $\nabla_{x}\log p(x)$ 
$$
x_{i+1} \leftarrow x_{i}+ \epsilon\nabla_{x} \log p(x) + \sqrt{2\epsilon}z_{i} \hspace{2em}i = 0,1,...,K
$$
where $z_{i} \sim N(0,I)$ when $\epsilon \rightarrow 0$ and $K \rightarrow \infty$ 

$x_{k}$ obtained from above markov chain will converge to a sample from $p(x)$ under some regularity conditions





***pitfalls***

estimated score functions are inaccurate in low density regions

*solutions*:

we can perturb data points with noise and train score-based models on the noisy data points instead.

how to determined the appropriate noise scale for perturbation process?

use the multiple scales of noise perturbations simultaneously  and training a **Noise Conditional Score-Based Models $s_{\theta}(x,i)$ (NCSN)**

objective:
$$
\sum_{i=1}^{L} \lambda(i)E_{p_{\sigma_{i}(x)}}[||\nabla_{x}\log p_{\sigma_{i}}(x)-s_{\theta}(x,i)||_{2}^{2}]
$$
$\lambda(i) \in R$ often chosen  to be $\lambda(i) = \sigma_{i}^{2}$


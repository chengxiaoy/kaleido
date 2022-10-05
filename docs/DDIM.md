## Denoising Diffusion Implicit Models

DDIM will be beneficial in the regime when sampling steps less than 50, while in 1000 step the DDPM still outperform.

> DDIMs, a more efficient class of iterative implicit probabilistic models with same training procedure as DDPMs
> generalize DDPMs via a class of non-Mavkovian processes that lead to the same training objective.

for iterative generative models, DDPM or NCSN have demonstrated the ability to produce samples comparable to that of GANs,
to achieve this, denoising autoencoder models are trained to denoise samples corrupted by various levels of Gaussian noise.

drawbacks of these models is that they required many iterations to produce a high quality sample.

benefits of DDIMs:
1. efficient sampling procedure
2. consistency, the same initial latent variable with various lengths of Markov Chain will produce similar high-level features
3. perform semantically meaningful image interpolation by manipulating the initial latent variable in DDIMs

as demonstrated in DDPMs:







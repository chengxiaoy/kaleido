## Hierarchical Text-Conditional Image Generation with CLIP Latents

> a two-stage model, a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditional on the image embedding.
>
> explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity.
>
> decoders conditional on the image representations can also produce variations of an image that preserve both its semantics and style. while varying the non-essential details absent from the image representation.
>
> using diffusion model for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.



### Method 



- A prior $p(z_{i}|y)$ that produces CLIP image embeddings $z_{i}$ conditioned on captions y. 
- A decoder $P(x|z_{i},y)$ that produces images $x$ conditioned on CLIP image embeddings $z_{i}$ (and optionally text captions $y$).

#### **Train a diffusion decoder to invert the CLIP image encoder.**



the presence of an encoder and its approximate inverse allows for capabilities beyond text-to-image translation.

*one notable advantage of using the CLIP latent space is the ability to semantically modify images by moving in the direction of any encoder text vector.* 

while there is much manual examination in GAN space.

![image-20221101104723034](pic\DALLE-2_1.png)

#### **who to train this encoder?**

modify the architecture used in GLIDE by projecting and adding CLIP image embeddings to the existing time step embedding, and by projecting CLIP embedding into four extra tokens of context that are concatenated to the sequence of outputs from the GUIDE text encoder.



past work using diffusion models show using guidance on the conditioning information improves sample quality a lot. so **using classifier-free guidance** by randomly setting the CLIP embeddings to zero 10% of the time, and randomly dropping the text caption 50% of the time during training.



**what's the classifier-free guidance?**

a form of guidance that interpolates between predictions from a diffusion model with and without labels.

In GANs and flow-based models, truncated or low-temperature sampling could attain a trade-off between image quality and diversity. using classifier guidance could obtained similar effect.

using a single neural network to parameterized both models(unconditional denoising model diffusion & conditional model), for unconditional denoising model we can simply input zeros for the classifier c when predicting the score.

perform sampling using following linear combination of the conditional and unconditional score estimates:
$$
\widetilde\epsilon_{\theta}(z,c) = (1+w)\epsilon_{\theta}(z,c) - w\epsilon_{\theta}(z)
$$
this is inspired from implicit classifier $p(c|z) \propto p(z|c)/p(z)$ 



in **GUIDE**, *classifier-free guidance is demonstrated that it's better than classifier guidance.*



#### Train a Prior

decoder can invert CLIP image embedding $z_{i}$ to produce images $x$, we need a prior model that produces $z_{i}$ from captions $y$ to enable image generations from text captions.

two different model classes for the prior model:

- **Autoregressive prior**:  the CLIP image embedding $z_{i}$ is converted into a sequence of discrete codes and predicted autoregressively conditioned on the caption $y$

- **Diffusion prior**: The continuous vector $z_{i}$ is directly modeled using a Gaussian diffusion model conditioned on the caption $y$.

using classifier-free guidance for both the *AR* and *diffusion prior*, by randomly dropping this text conditioning information 10%.



the ***Diffusion prior*** architecture:

train a decoder-only Transform with a causal attention mask on a sequence consisting of, in order: 

encoded text, the CLIP text embedding, an embedding for the diffusion timestep, the noised CLIP image embedding and a final embedding whose output from the Transformer is used to predict the unnoised CLIP image embedding.

 Instead of using the $\epsilon$-prediction formulation, we find it better to train our model to predict the unnoised $z_{i}$ directly, and use a mean-squared error loss on this prediction.



### Image Manipulations

using DDIM we can get the bipartite representation $(z_{i}, x_{T})$ that is sufficient for the decoder to produce an accurate reconstruction. the former is obtained by CLIP image encoder, the latter is obtained by applying DDIM inversion to $x$ using the decoder. 



**Variation**

by changing the $\eta$ in  DDIM we can get variations of the original image. As $\eta$ increases, these variations tell us what information was captured in the CLIP image embedding (and thus is preserved across samples), and what was lost (and thus changes across the samples).

**Interpolations**

rotate between CLIP embeddings $z_{i1}$ and $z_{i2}$ using spherical interpolation $z_{i\theta} = slerp(z_{i1},z_{i2},\theta)$

**Text Diffs**

compute a text diff vector $z_{d} = norm(z_{t}-z_{t_{0}})$ and yielding intermediate CLIP representations $z_{\theta} = slerp(z_{i},z_{d},\theta)$ where $\theta$ is increased linearly from 0 to a maximum value that is typically in [0.25, 0.5]


### Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation

**Abstract**

> Our pSp framework is based on a novel encoder network that directly generates a series of style vectors which are fed into a pretrained StyleGAN generator, forming the extended W+ latent space. We first show that our encoder can directly embed real images into W+, with no additional optimization.

`utilizing encoder to directly solve image-to-image translation tasks, defining them as encoding problems from input domain into the latent domain.`



*Two Questions:*

***what's the definition of the $w$+ space, and the difference between $w$+ and $w$ in StyleGAN?***

***who to generates style without additional optimization?***

_as inverting a real image into a 512-dimensional vector w does not lead to an accurate reconstruction.
Motivated by this, it has become common practice to encode real images into an extended latent space, W+, 
defined by the concatenation of 18 different 512-dimensional w vectors, one for each input layer of StyleGAN._ 

These works usually resort to using per-image optimization over W+, a fast and accurate inversion of real image into W+ 
is a challenge.









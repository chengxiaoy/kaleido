## StyleGAN2

> expose and analyze several of stylegan's characteristic artifacts,
> and propose changes in both model architecture and training methods to address them.


- redesign the generator normalization
- revisit progressive growing
- regularize the generator to encourage good conditioning in the mapping from latent codes to images


#### characteristic artifacts
> artifact is puzzling, as the discriminator should be able to find it.   
> We pinpoint the problem to the AdaIN operation that 
> normalizes the mean and variance of each feature map separately, thereby potentially destroying any information found
> in the magnitudes of the features relative to each other.


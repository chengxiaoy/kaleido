### Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors



##### Image Synthesis method:

- **GAN** generative adversarial network
- **VAE** variational auto encoder
- **GLO** latent embedding learning 
- **IMLE** nearest-neighbor based implicit maximum likelihood estimation
- **AR** auto regression model



this paper proposed a novel method -- Generative Latent Nearest Neighbors **GLANN**

##### About GAN

three domain that GAN mainly used to:

1. trainin effective unconditional image genrators
2. almost the only method for unsupervised image translation between domains( but NAM)
3. being an effective perceptual image loss function (eg. Pix2Pix)



disadvantages:

1. hard to train

2. mode collapse

   

#####  About GLO

> embeds the training images in a low dimensional space, so that they are reconstructed when the embedding is passed through a jointly trained deep generator. The advantages of GLO are i) encoding the entire distribution without mode dropping ii) the learned latent space corresponds to semantic image properties 

the critical disadvantage of GLO is that there is not a principled way to sample new images from it.



**so what's the different between GLO and VAE?**

in trainin phase:

1. constraining all latent vectors to lie on a unit sphere or a unit ball
2. using a Laplacian pyramid loss (VGG perceptual loss is works better)
3. **using latent optimization** 

the reconstruction iamge of GLO is best as the latent code is free during the training process. and for a more complex distribution the  GLO and VAE are generate poorly while the GAN work well. ***LSUN dataset***

##### About IMLE

> training generative models by sampling a large number of latent codes from an arbitrary distribution, mapping each to the image domain using a trained generator and ensuring that for every training image there exists a generated image which is near to it. 

IMLE is easy to sample and do not suffer from mode-collapse, while the metric in pixel space limit it to only can generate blurry images 

the IMLE  procedure:

![IMLE](/home/yons/PycharmProjects/ethan-vae/docs/pic/IMLE.png)



**About GLANN**



> Our method overcomes the metric problem of IMLE by first embedding the training images using GLO. The attractive linear properties of the latent space induced by GLO, allow the Euclidean metric to be semantically meaningful in the latent space Z. We train an IMLE-based model to map between an arbitrary noise distribution E, and the GLO latent space Z. The GLO generator can then map the generated latent codes to pixel space, thus generating an image.


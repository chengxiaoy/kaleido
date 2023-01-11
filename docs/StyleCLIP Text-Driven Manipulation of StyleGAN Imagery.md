## StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery

> explore leveraging the power of recently introduced Contrastive Language-Image Pre-training (CLIP) models in order to develop a text-based interface for StyleGAN image manipulation that does not require such manual effort.





*this paper will introduce three methods, the performance in each phase are shown as below:*

![image-20221031101749651](pic\StyleCLIP_1.png)



### Latent Optimization based on CLIP Loss

given a source latent code $w_{s} \in W+$ and a *text prompt t*, solve the following optimization problem:
$$
agrmin_{w\in W+}D_{CLIP}(G(w),t)+\lambda_{L2}||w-w_{s}||+\lambda_{ID}L_{ID}(w)
$$
where G is a pretrained StyleGAN generator and $D_{CLIP}$ is the cosine distance between CLIP embeddings of its two arguments.
$$
L_{ID}(w) = 1-\langle R (G(w_{s})),R(G(w))\rangle
$$
where $R$ is a pretrained ArcFace network for face recognition 

### Latent Mapper

![image-20221031105043702](pic\StyleCLIP_2.png)

the Mapper is defined by:
$$
M_{t}(w) = (M_{t}^{c}(w_{c}),M_{t}^{m}(w_{m}),M_{t}^{f}(w_{f}))
$$
the mapper is trained to manipulate the desired attributes of the image as indicated by the text prompt t:
$$
L_{CLIP}(w) = D_{CLIP}(G(w+M_{t}(w)),t) \newline L(w) = L_{CLIP}(w) +\lambda_{L2} \|M_{t}(w)\| + \lambda_{ID}L_{ID}(w)
$$




the direction of the step in the latent space varies over different inputs:

![image-20221031110114914](pic\StyleCLIP_3.png)



### Global Directions

latent mapper sometimes falls short when a fine-grained disentangled manipulation is desired.

**propose a method for mapping a text prompt into a single, global direction in StyleGAN's style space $S$**, **obtain a vector $\Delta t$ in CLIP's joint language-image embedding and then map this vector into a manipulation direction $\Delta s $ in $S$** 

A stable $\Delta t$ is obtained form natural language using prompt engineering.

The corresponding direction $\Delta s$ is then determined by assessing the relevance of each style channel to the target attribute.



**prompt engineering to get $\Delta t$**

in this method, text description of a target attribute and a corresponding neutral class should be both provided. Prompt engineering will be applied to produce the normalized average embeddings, and the difference is used as the target direction $\Delta t$



**Channelwise Relevance**

assess the relevance of each channel $c$ of $S$ to a given direction $\Delta i$ in CLIP's embedding space.

generate a collection of style code $s \in S$ and perturb only the $c$ channel of each style code by adding a negative or positive value. (actually 100 image pairs to estimate the mean)
$$
R_{c}(\Delta i) = E_{s \in S}\{\Delta i_{c} \cdot \Delta i\}
$$
Having estimated the relevance $R_{c}$ of each channel, we ignore channels whose $R_{c}$ falls below a threshold $\beta$



![image-20221031153714902](pic\StyleClip_4.png)

![image-20221031153813555](pic\StyleCLIP_5.png)








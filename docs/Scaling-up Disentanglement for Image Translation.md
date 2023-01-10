### Scaling-up Disentanglement for Image Translation



this paper is an extension of LORD which is proposed by the same author.

The author has observed that entanglement in image translation are common exist in previous model

 ![Entanglement in image translation](/home/yons/PycharmProjects/ethan-vae/docs/pic/LORD2_1.png)



so in this paper, avoid the previous assumption that is labeled attributes and unlabeled attributes are uncorrelated by assume the existence of correlated attributes.



the Attributes can be consists of three part:

1. Labeled Attributes $y$
2. Correlated Attributes 
3. Uncorrelated Attributes

$$
x_{i} = G(y_{i}, u_{i}^{corr},u_{i}^{uncorr})
$$

a Framework named **OverLORD** has been proposed which is composed by *disentanglement* and s*ynthe*

*sis*



![LORD2_2](/home/yons/PycharmProjects/ethan-vae/docs/pic/LORD2_2.png)



the Disentanglement stage using latent optimization to learn the embedding vectors.



in this framework, some method are introduced to learn some of the correlated attributes independently from the uncorrelated ones.

form a function $T$  that outputs an image $x^{corr} = T(x)$, which retains the correlated attributes but exhibits different uncorrelated attributes. and the correlated attributes are modeled by 
$$
u^{corr} = E_{c}(x^{corr}) = E_{c}(T(x))
$$
two different forms of $T$:

1. *Random spatial transformation*: making $x^{corr}$ retain attributes that are *pose-independent*

   ​	
   $$
   T(x) = (f_{crop} \circ f_{rotate} \circ f_{flip})(x)
   $$



2. *Masking*: if the correlated attributes are localized, we can set $T$ to mask-out the uncorrelated attributes, retaining only correlated regions
   $$
   T(x;m) = x\odot m
   $$



the loss in disentanglement stage should be consists of reconstruction loss and bottleneck loss:


$$
L_{rec} = \sum l(G(y_{i},u_{i}^{corr},u_{i}^{uncorr}),x_{i})
$$

$$
L_{b} = \sum ||u_{i}^{uncorr}||^2
$$



for reconstruction $l$ VGG-based preceptual loss is used.



after the disentanglement stage, two encoders are trained $E_{y}: X \rightarrow Y$ and $E_{u}: X \rightarrow U$



the loss should be 
$$
L_{enc} = \sum \| E_{y}(x_{i})-y_{i}\|^2 + \|E_{u}(x_{i})-u_{i}^{uncorr}\|^2 \\ 
L_{gen} = \sum l(G(E_{y}(x_{i}),E_{c}(x_{i}),E_{u}(x_{i})),x_{i}) \\
L_{adv} = \sum logD(x_{i})+log(1-D(\bar x_{i})
$$



















## VARIATIONAL LOSSY AUTOENCODER

站在表示学习的视角，学习适合下游任务的表示是主要任务，如果对于实际任务学习2D图片的全局结构并忽略局部细节是有用的。文章首次将VAE和neural autoregressive models 有效结合在一起,通过相应的结构设计可以使全局隐变量忽略相应的图片细节，以一种有损方式。同时利用autoregressive model拟合先验和decoding distribution $p(x|z)$ 可以达到当时的密度估计任务中的sota



使用生成模型学习表示有一些缺点：

对于观测数据，不同的生成模型使用不同的隐变量拟合相同的概率密度函数，因此我们的隐变量大概率由模型假设决定。同时优化经常和我们的目标无关，比如autoregressive model 可以很好拟合数据但是根本没有任何随机隐变量，因此限制了ar模型的流行度。



### VAE do not autoencode in General

在传统的图片领域 :

decoding distribution $p(x|z)$ is usually chosen to be a simple factorized distribution, i.e. $p(x|z)=\prod_{i}p(x_{i}|z)$ ,and this setup often yields a sharp decoding distribution $p(x|z)$ that tends to reconstruct original datapoint x exactly.



思路切换到 hybird model has both the vae and ar model. 以前为了将$p(x|z)$容量更大 使用了自回归的循环序列模型

conditions under which it is guaranteed to autoencode (reconstruction being close to original datapoint) 很少被讨论, 当解码器$p(x|z)$使用更加灵活强大的分布式的时候,隐变量不会被使用，除非通过dropout等技术将decoder weaken.

从Bits-Back Coding 和 信息选择角度分析 informantion preference:  

从 $p(x)=\int_{z}p(z)p(x|z)dz$ 角度分析 更加强大的$p(x|z)$ 有利于$p(x)$的建模，但是当吧vae和sequence model结合在一起时



当decoder采用RNN  with autoregression dependency 的时候, the RNN autoregressive decoding can in theory represent any probability distribution even without dependence on z. the latent code z is ignored. 因为这个后验 $q(z|x)$ carries little informantion about datapoint x.  it is easy fot the model just set the approximate posterior to be the prior to avoid paying any regularization cost $D_{KL}(q(z|x)||p(z))$



文章第一个观点：

从bits-back coding perspective of VAE 理解当deconding  distribution sufficiently powerful and intractable true posterior distributions 时会出先这种现象

<!--*但是这个论点有点靠不住，因为就算decoder的能力很强时可以represent $p(x)$， 如果近似后验和先验一致，也无法将objective减少到最低，因为objective是要求重构而不是拟合$p(x)$--> 



同时文章第二个观点：

用标准流要求先验的效果 和使用复杂的近似后验效果是一致的



文章在第一个观点的基础上提出了有选择性的coding，如果decoder可以自动学习texture 那么$z$ 中将包含全局信息, 因此在信息上是lossy的， 所以也叫lossy variational autoencoder

第二个观点是 对先验使用标准流拟合复杂分布效果和对后验使用标准流效果是一致的









 

















 


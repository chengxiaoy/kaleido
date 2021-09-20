### 什么是VAE



How can we perform efficient approximate inference and learning with directed probabilistic models whose continuous latent variables and/or parameters have intractable posterior distributions?

在概率图模型中，我们虽然可以利用图结构来表达变量之间的条件独立性从而简化运算，但是依然有很多实际问题困扰着模型的学习和推断。如 变量之间的关系复杂不能由先验信息确定图结构时，需要进行structure learning，对于变量是真实情况下的连续值，变量之间的条件概率是复杂多峰时，难以学习参数。

当然第一步需要清楚 深度学习中的概率图模型和传统的PGM之间的差异（bilibili视频）

The variational Bayesian (VB) approach involves the optimization of an approximation to the intractable posterior

变分贝叶斯 提供了一种近似难以解决的后验的优化方法.



#### 问题描述 problem scenario

假设数据集$X$由独立同分布的连续或离散样本$x_i$组成，我们假设这些数据都是通过随机过程产生，其中涉及到了连续随机变量$z$。过程由两部分组成：

1. $z_i$取自于一个先验分布$p_{\theta^*}(z)$
2. 样本$x_i$从条件分布$p_{\theta^*}(x|z)$中采样

我们再次假设先验$p_{\theta^*}(z)$和$p_{\theta^*}(x|z)$来自同一参数族，同时他们的概率密度函数（PDF）对于$\theta$和$z$在任意位置是可微分的。

对于假设的过程，真实的参数$\theta^*$和隐变量$z_i$都是未知的。

通常我们会通过假设边缘或者后验概率来简化运算，但是vae算法在以下情况下依然可以有效运算：

1. 计算复杂: 对于边缘似然$P_\theta(x) = \int{p_\theta(z)p_\theta(x|z)}dz$难以计算(cannot evaluate or differentiate the marginal likelihood)，后验密度$p_\theta(z|x) = p_\theta(x|z)p_\theta(z)/p_\theta(x)$ 难以计算(so the EM algorithm cannot be used)， 积分的存在对于任何平均场变分贝叶斯算法也是难以处理的。这种难以计算性在处理适当复杂的似然函数$p_\theta(x|z)$是普遍存在的。
2. 数据集大：批量更新耗费太高，我们更愿意使用小批量甚至是单个样本点，而基于采样的方法例如Monto Carlo EM 运行时间太慢因为在每一个数据点上都进行采样循环。

而本文提出的方法可以解决上述场景中的三个问题：

1. 有效的对$\theta$近视ML和MAP估计，对$\theta$感兴趣是因为我们可以模拟随机过程并生成类似真实数据的人造数据。
2. 在给定$x$和$\theta$时有效的对于隐变量$z$的近似后验推断，这有助于编码或者是表示学习的任务。
3. 有效的对于x的近似边缘推断，有助于任何需要x先验的推断任务。通用的应用在计算机视觉领域包含图像去噪、修复和超分辨率。



#### the variational bound

边缘似然由单个数据点的似然的和组成，即：
$$
logp_\theta(x^{(1)},...,x^{(N)}) = \sum_{i=1}^{N}logp_{\theta}(x^{(i)})
$$
又可以被重写为：
$$
logp_{\theta}(x^{(i)}) = D_{KL}(q_\phi(z|x^{(i)})||p_\theta(z|x^{(i)}))+\mathcal{L}(\theta,\phi;x^{(i)})
$$
因为KL-散度为非负的：
$$
logp_{\theta}(x^{(i)}) > \mathcal{L}(\theta,\phi;x^{(i)}) = E_{q_\phi(z|x)}[-logq_{\phi}(z|x)+logp_\theta(x,z)]
$$
也可以表示为：
$$
\mathcal{L}(\theta,\phi;x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)})||p_\theta(z)) + E_{q_\phi(z|x^{(i)})}[logp_\theta(x^{(i)}|z)]
$$
当我们优化下界$\mathcal{L}(\theta,\phi;x^{(i)})$ 时需要对$\phi$和$\theta$微分，但是使用常用的Monto Carlo gradient estimator
$$
\nabla_{\phi}E_{q_{\phi}(z)}[f(z)] = E_{q_{\phi}(z)}[f(z)\nabla_{q_{\phi}(z)}logq_{\phi}(z)] \simeq \frac{1}{L}\sum_{l=1}^{l}f(z)\nabla_{q_{\phi}(z^{(l)})}logq_{\phi}(z^{(l)})
$$
估计时会产生较大方差 





#### 参考

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

[β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK](https://openreview.net/references/pdf?id=Sy2fzU9gl)

[Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)


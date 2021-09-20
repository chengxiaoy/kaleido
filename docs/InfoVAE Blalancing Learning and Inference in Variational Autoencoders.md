### InfoVAE: Blalancing Learning and Inference in Variational Autoencoders



论文主要平衡 learning和inference 两部分的objective

$$
\mathcal{L}_{ELBO} = -D_{KL}(q_{\phi}(x,z)||p_{\theta}(x,z))\\\
=-D_{KL}(p_{\mathcal{D}}(x)||p_{\theta}(x)) - E_{p_{\mathcal{D}}(x)}[D_{KL}(q_{\phi}(z|x)||p_{\theta}(z|x))]\\\
=-D_{KL}(q_{\phi}(z)||p(z)) - E_{q_{\phi}(z)}[D_{KL}(q_{\phi}(x|z)||p_{\theta}(x|z))]
$$
主要观点如下:

1. Good ELBO values  do not imply accurate inference 
2. Implicit modeling bias (x is often higher dimensional compared to z)


$$
\hspace{-3cm}\mathcal{L}_{ELBO}(x) = \mathcal{L}_{AE}(x) + \mathcal{L}_{REG}(x)\\
\hspace{4cm} \equiv E_{q_{\phi}(z|x)}[logp_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x)||p(z))
$$
$\mathcal{L}_{AE}(x)$ 鼓励$q_{\phi}(z|x)$ 不相交当$X_{i} \neq X_{j}$

而在实际的使用中, 变分分布族 $q_{\phi}$ 一般在整个隐变量空间中阻止 disjoint support

学习$\mathcal{L}_{AE}(x)$ 的过程会要求mass of the distributions away from each other

但是$\mathcal{L}_{REG}(x)$ 并不能 counter-balance this tendency



同时 $\mathcal{X}$ 维度一般比$\mathcal{Z}$高,所以任何$\mathcal{X}$ 空间的错误都会比$\mathcal{Z}$空间中的错误高数量级

例如
$$
D_{KL}(\mathcal{N}(0,I),\mathcal{N}(\epsilon,I)) = n\epsilon^{2}/2
$$
n是分布的维度,所以当 z和x起冲突的时候,模型偏向与牺牲z而满足x



从信息选择属性上讲 The Information Preference Property

PixelRNN/PixelCNN 可以有效提高图像质量,但是通常会忽略z, 也即是z和x之间的mutual information 互信息变得很小。而学习和输入相关的隐变量是无监督学习的任务之一。

从信息角度来讲 我们可以找到一个和x无关的z，对于任何z， $P_{\theta}(x_{i}|z)$ 都和$P_{D}(x_{i})$ 一致，

而 z和x无关后，$p_{\theta}(z|x)$ = $p(z)$，所以很容易找到$\phi$使对于任意x $q_{\phi}(z|x)$ = $p(z)$

从而开篇的ELBO可以被优化到最大值0



为了提高模型对z变量推断的准确性 提高在objective中的权重，INFOVAE 提出了新的objective
$$
\mathcal{L}_{InfoVAE} = -{\lambda}D_{KL}(q_{\phi}(z)||p(z)) - E_{q(z)}[D_{KL}(q_{\phi}(x|z)||p_{\theta}(x|z))] + {\alpha}I_{q}(x;z)
$$

1. 提高z变量的约束
2. 增加x和z的互信息量项

上述等式不能直接优化，重写为：
$$
\mathcal{L}_{InfoVAE} = E_{p_{D}(x)}E_{q_{\phi}(z|x)}[log_{p_{\theta}(x|z)}]-\\\
(1-{\alpha})E_{p_{D}(x)}D_{KL}(q_{\phi}(z|x)||p(z))-\\\
(\alpha +\lambda-1)D_{KL}(q_{\phi}(z)||p(z))
$$
前两项与传统的变分编码器无异，
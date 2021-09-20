## Neural Discrete Representation Learning

本文提出的VQ-VAE 和传统的VAE有两大不同：

1. 编码器输出离散的编码
2. 隐变量的先验分布不再是静态的，而是可被学习的

语言是离散的，而图片可以被语言描述 (image capture)， 因此某种程度上说用离散的隐变量来解释图片是有道理的，而在深度学习中，威力强大的autoregressive model 被用来拟合离散变量的分布。

基于向量量化的方法可以避免大variance和“posterior collapse”（decoder 过于强大会忽略latent，infoVAE中有相关解释）

由于隐变量的分布不再是静态的固定分布，所以对于encoder的梯度 使用类似straight-through estimator的方法将decoder的input的梯度拷贝给后验隐变量。

整个模型的目标由三个部分组成
$$
\mathcal{L} = logp(x|z_{q}(x))+||sg[z_{e}(x)]-e||_{2}^{2}+\beta||[z_{e}(x)-sg[2]||_{2}^{2}
$$


由于 $z_{q}(x)$ 的取值为
$$
z_{q}(x) = e_{k}, \hspace{0.5cm}where\hspace{0.3cm}k = argmin_{j}||z_{e}(x)-e_{j}||^{2}
$$
因此并没有真实的梯度，文章使用近视梯度将$z_{q}(x)$的梯度直接赋给了$x$  类似straight-throught estimator的处理方式，在pytorch代码中为自定义functional 实现

由于e 是离散的，而loss 的第二项就是为了学习离散空间中的向量分布，同时第三项为了稳定x的离散表达。



对于latent的先验prior，文章使用auto-regressive 的分布进行建模，对于图片使用PixelCNN 对离散的隐变量进行建模，而针对音频使用WaveNet
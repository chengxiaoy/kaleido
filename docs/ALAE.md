### Adversarial Latent Autoencoders



首先 文章先讨论了 之前VAE和GAN 在框架上的结合

为什么要互相结合？因为有各自的优缺点 而优缺点是在算法的假设和结构上决定的

VAE 的优点:

- 训练稳定 （GAN训练不稳定）
- 不会出现 mode collapse( GAN 常见的问题 需要使用trick缓解) 因为基于最大似然的学习
- 结构上支持 input到latent code 的推导

缺点

- 图片模糊 没有GAN生产的图片那么锐利

同时disentanglement 也是在VAE 领域研究的热点



VAE 在假设上的缺点是 

> the latent space should have a probability distribution that is fixed a priori and the autoencoder should match it.

虽然在 VAE的tutorial 上面 有相关解释 为何可以将隐空间 看成是高斯分布或者某一个特定分布,但是 却无法保证 我们训练得到的decoder 是match 这个impose 隐空间的，并且对于一个impose的空间 disentanglement 也是更加困难的








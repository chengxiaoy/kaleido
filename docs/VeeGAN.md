## VeeGAN  https://akashgit.github.io/VEEGAN/

VeeGAN 主要聚焦在 减少GAN mode collapse 的目标中

核心思想是 因为 generator 在z->x的映射中会出现mode collapse 的问题  那么会出现 多个 不同z 映射成了相同的x  如果可以将 x->z 映射成 多种值的z  那么就可以减少多样性减少的问题

核心原因是 z作为noise 的变量很好判断多样性 可以作为一种reconstruct loss

跟之前带resconstructor network的 BiGAN 和ALI 相比的话 noise autodecoder 可以明显地减少mode collapse

和其他在data space 里面做reciprocity 不同 VeeGAN  对noise 进行编码思想是 在低维的噪声里面noise 更有意义



背景：

隐概率分布 被 一个采样器所定义 并没有一个清晰易解的概率密度分布 GAN 就是一种隐概率模型



从理论推导上有点复杂  也不是很strong 把真是的p(x)当成先验 通过GAN 去逼近

关键点在于 在ALI 或者BiGAN的基础上 添加了reconstruction loss 去防止mode collapse 



别切这个reconstruction loss 是在 latent space 中构建的 相比 data space 更加有效
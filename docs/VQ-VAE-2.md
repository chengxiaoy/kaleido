## vector quantized variational autoencoder(VQ-VAE-2 )

VQ-VAE 只需要在压缩的隐变量中使用auto regression 模型进行采样，与在像素空间中进行采样 毫无疑问会快很多，同时使用了一个多尺度的层次结构，可以达到目前GAN的最先进的高保真程度，并且没有GAN的缺点：model collapse和lack of diversity

深度生成模型在以下领域得到了快速应用，高分辨率 super resolution, domain editing, artistic manipulation, text to speech and music generation

生成模型有两大类

1. likelihood based models, VAE, flow based, autoregression mdel
2. implicit generative models, such as GAN

GAN 使用了minimax objective， 可以生成 high-quality high-resolution 的图片，但是没有捕捉到数据的真实分布生成的数据缺乏多样性，同事GAN生成的图片evaluation较为复杂一般使用IS 和FID

而基于最大化似然的方法理论上可以捕捉到训练数据的所有分布，但是在像素空间中直接优化最大化似然是很有挑战的。

shortcoming如下：

1. 在像素空间中的NLL negative log likelihood 无法衡量图片样品的quality
2. 不同模型之间不能使用NLL比较（应该是不同模型之间对于概率分布 先验、重构的假设不同）

VQ-VAE使用有损压缩的思想减轻模型对于不重要信息的建模，将图片压缩到离散的隐变量空间通过自动编码器，这种离散的表示可以使用现在最先进的PixelCNN使用了self-attention的PixelSnail 来建模作为先验。



使用深度模型对隐变量的分布进行建模已经是提高图片质量的共识，而VQ-VAE优点包括 使用威力强大的ar 自回归模型和two stage

top latent map 可以拟合全局信息，装备multi-headed self-attention layers可以从更大的感受视野中获取距离较远部分之间的correlation信息，与之相比bottom的在top latent中的条件先验有助于high resolution。

MLE obejctive can be expressed as the forward KL divergence between the data and the model vq vae的核心思想 latent不再是连续的 而是discrete vq即是vector quantized 的意思

文章代码部分的亮点是：

1. 与VQVAE相比，使用了更为强大的先验 multi-header and self-attention
2. 在隐变量的建模是上使用了 层次结构，上层隐变量学习全局信息，而下层隐变量学习局部信息
3. condiPixelCNN 是怎么操作的？





从pixelcnn的角度看到话：

autoregression 可以达到当前的最好的密度估计的水平 但是却无法抽取latent 隐变量

所以 当ar应用到latent的priori先验的时候，可以不再要求$p(z)$是高斯分布 

pixel cnn 针对离散变量直接通过output输出离散值的多项式分布

但是依然没有解释为啥vq-vae为啥图片生成的这么清楚, 














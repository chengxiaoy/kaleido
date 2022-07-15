## A Connection between Score Matching and Denoising Autoencoders



#### Score Matching

Score Matching was introduced as a technique to learn the parameters $\theta$ of probablility density models $p(x;\theta)$ with intractable partition function $Z(\theta)$, where $p$ can be written as
$$
p(x;\theta) = \frac{1}{Z(\theta)} \exp(-E(x;\theta))
$$
we will call ***score*** the gradient of the log density with respect to the data vector: $\psi(x;\theta) = \frac{\partial \log p(x;\theta)}{\partial x}$

we are talking about a ***score*** with respect to the ***data***



  

**Explicit Score Matching** (**ESM**)

***optimization objectives***
$$
J_{ESM_{q}} = E_{q(x)}[\frac{1}{2}\left\|\psi(x;\theta)-\frac{\partial \log q(x)}{\partial x}\right\|^{2}]
$$
**Implicit Score Matching**(**ISM**)



![ISM](C:\Users\chengxiaoy\PycharmProjects\kaleido\docs\pic\ISM.png)

where $\psi_{i}(x;\theta) = \psi(x;\theta)_{i} = \frac{\partial\log p(x;\theta)}{\partial x_{i}}$  $C_{1}$ is a constant that does not depend on $\theta$



**Denoising Score Matching**(**DSM**)


$$
J_{DSM_{q_{\sigma}}}(\theta) = E_{q_{\sigma}(x,\tilde x)}[\frac {1}{2} \left\| \psi(\tilde x; \theta)-\frac {\partial \log q_{\sigma}(\tilde x|x)}{\partial \tilde x} \right\|^{2}]
$$


underlying intuition is that following the gradient $\psi$ of the log density at some corrupted point $\tilde x$ should ideally move us towards the clean sample $x$ 



considered Gaussian kernel we have
$$
\frac {\partial \log q_{\sigma}(\tilde x| x)}{\partial \tilde x} =  \frac {1}{\sigma ^2}(x - \tilde x)
$$
direction $\frac {1}{\sigma^{2}}(x-\tilde x)$ clearly corresponds to moving from $\tilde x$ back to $x$



proof show training denoising autoencoder is equivalent to performing score matching with the energy function of some formula on Parzen density estimate:
$$
J_{ESM} = J_{DSM} = J_{DAE}
$$

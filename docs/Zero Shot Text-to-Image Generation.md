## Zero Shot Text-to-Image Generation

> a simple large-scale generative model, approach based on a transformer that 
> autoregressively models the text and image tokens as a single stream of data

### Method:



#### Stage 1 (train a variational auto-encoder to compress image to token)
train a discrete VAE to compress each 256×256 RGB image to 32×32 grid of 
image tokens, each element of which can assume 8192 possible values.

tips:
the initial prior of z is uniform categorical distribution over the K = 8192 
codebook vectors.    

as the q is a discrete distribution, reparameterization trick is not suitable.
use the **gumbel-softmax relaxation** and the likelihood of x is evaluated using 
the log-laplace distribution.

#### stage 2 learning the prior
Given a text-image pair, BPE-encode the lowercased caption using a most 256 tokens
with vocabulary size 16384 and encode the image using 32×32 tokens with vocabulary
size 8192. the image tokens are obtained using argmax sampling from the dVAE

tips:
learn a special padding token for the padding position in the text, this result 
in higher validation loss, but better performance on out-of-distribution captions


### Mixed-Precision Training

To save GPU memory for this 12 billion parameters model, most parameters are stored in 16-bit
precision. Training in 16-bit precision without diverging was the most challenging part of this project.

the root cause of instability is the underflow in the 16-bit gradients. a set guidelines:
1. use pre-resblock gradient scaling instead of standard loss scaling.
2. avoid underflow when dividing the gradient.
3. only use 16-bit precision where it is really necessary for performance.

### Sample Generation

Using a pretrained contrastive model to assign a score to caption-image pair based on how well 
the image matches the caption.
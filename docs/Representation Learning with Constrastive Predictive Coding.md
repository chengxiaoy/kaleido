## Representation Learning with Contrastive Predictive Coding (CPC)

> The key insight of our model is to learn such representation by predicating
> the future in latent space by using powerful autoregressive models, which we call
> **Contrastive Predicating Coding**, this is a universal unsupervised learning approach
> to extract useful representations.

#### Methods:
1. compress high-dimensional data into a much more compact latent embedding 
space in which conditional predictions are easier to model.
2. using autoregressive models in this space to make predictions many steps in the future.
3. rely on `Noise-Contrastive Estimation` for the loss function

#### Motivation and Intuitions
Approaches that use next step prediction exploit the local smoothness of the signal. 
When predication further in the future, the amount of shared information becomes much lower,
and the model needs to infer more global structure.

unimodal losses such as meansquared error and cross-entropy are not very useful

#### Architecture
![cpc model](pic/cpc.png)

> we do not predict future observations $x_{t+k}$ directly with a generative model
> instead we model a density ratio which preserves the mutual information between x_{t+k} and c_{t}
> Both the encoder and autoregressive model are trained to jointly optimize a loss based on NCE


### Loss

from the reader's perspective, this unsupervised method learn representation by predicting the future, the key method is
called infoNCE, the loss is the categorical cross-entropy of classifying the positive sample correctly.
![infoNCE](pic/infoNCE_loss.PNG)

### Experiment 

> from a 256x256 image we extract a 7x7 grid of 64x64 crops with 32 pixels overlap.

![cpc_vision](pic/cpc_vision.png)

## Learning Representations by Maximizing Mutual Information Across Views



> an approach to self-supervised representation learning based on maximizing mutual information between features extracted from multiple views of a shared context.

how to **maximizing** mutual information? how to define **mutual information**?



>  Maximizing mutual information between features extracted from these views requires capturing information about high-level factors whose influence spans multiple views



the proposed model is based on **local Deep InfoMax DIM**

extends the DIM in three key ways:

1. predict features across independently-augmented versions of each input
2. predict features simultaneously across multiple scale
3. a more powerful encoder

> Predicting across independently-augmented copies of an input and predicting at multiple scales are two simple ways of producing multiple views of the context provided by a single image.

*the DIM paper: LEARNING DEEP REPRESENTATIONS BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION*



Local DIM seeks an encoder $f$ that maximizes the mutual information 


$$
I(f_{1}(x);f_{7}(x)_{ij})\hspace{1em} in \hspace{1em} p(f_{1}(x),f_{7}(x)_{ij})
$$



the best result with local DIM were obtain using a mutual information bound based on **Noise Contrastive Estimation (NCE)** 

we can maximize the **NCE** lower bound by minimizing the following loss:
$$
E_{(f_{1}(x),f_{7}(x)_{i,j})}[E_{N_{7}}[L_{\phi}(f_1(x),f_7(x)_{ij},N_7)]]
$$
the task of the antecedent feature is to pick its true consequent out of a large bag of distractors(negative pair).



**Augmented Multiscale Deep InfoMax (AMDIM)**



![image-20211226212453399](pic\AMDIM.png)

if the receptive fields for features in a positive sample pair overlap too much, the task becomes too easy and the model performs worse.



the results is amazing!

![image-20211226215316501](pic\AMDIM_SCORE.png)

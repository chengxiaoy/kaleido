### DreamBooth && Textual Inversion

#### Textual Inversion

> a textual inversion method that learns to represent visual concepts though new pseudo-words in the embedding space of a frozen text-to-image model, that's say it just fine-tune the text encode embeddings for the new tokens, which is still constraint by the original output domain

#### DreamBooth

> fine-tune a specific model  such that it learns to bind a unique identifier with that specific subject, the text-to-image is fine-tuned in order to embed the subject within the output domain of the model, enabling the generation of novel images of the subject while preserving key visual features that form its identity.



both these two papers are focus on how to create a particular concept from a specific subject.  



the difference of these two model is that the former only optimize the embedding in the text encoder, while DreamBooth will optimize the U-net model meanwhile.





for Textual Inversion, the model aim to learn the embedding of a new token which represent the visual concept

![Textual Inversion_1](/home/yons/PycharmProjects/ethan-vae/docs/pic/Textual_Inversion_1.png)

as shown in the figure, the model optimize the embedding vector $v_{*}$ by the reconstruction objective.



the concept could be a style or object, when the concept is style we could use the concept for style-guide generation.

![Textual Inversion Style](/home/yons/PycharmProjects/ethan-vae/docs/pic/Textual_Inversion_2.png)
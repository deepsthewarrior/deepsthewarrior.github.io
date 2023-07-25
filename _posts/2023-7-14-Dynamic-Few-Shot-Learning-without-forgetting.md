
The paper quotes on 2 contributions. It highlights on not forgetting the base classes when learning on the novel categories.
* attention based few-shot classification weight generator
* convert the ConvNet as the cosine similarity function between input features and the weight vectors.

#### ATTENTION BASED WEIGHT VECTOR GENERATION:
When a novel class is seen, with the input features of the novel class, the model generates a seperate branch with weight vectors for the novel class. These weight vectors are created based on the existing visual information from the weight vectors of the base classes, using attention mechanism. 

Attention mechanism helps in obtaining better performance when the novel class samples are very few like one sample.

#### COSINE SIMILARITY BASED CONVNET RECOGNITION MODEL:
The weight vectors from the previous contribution cannot be added to the classifier, the base class weights and the novel class weights are learned with different momentum in SGD. Thus the logits are not on the same scale, might not allow the learning of different categories in a uniform manner.   Hence, the classification scores is calculated as the cosine similarity function between input features and the generated weight vectors.

<figure style="text-align:center;">
  <img src="/images/images_1.PNG" alt="Result" />
  <p class="img-caption"> Architecture (Source: <a href="https://arxiv.org/pdf/1804.09458.pdf">https://arxiv.org/pdf/1804.09458.pdf</a>)</p>
</figure>


#### Why does cosine similarity based classifier work?
Cosine similarity is calculated between the normalized input feature vector vs normalized weight vectors. In order to minimize the loss, the feature vector must be identical to the weights of the ground truth category. 
This generalizes the L2 normalized feature vectors, and reduces intraclass variance, as the feature vector should be able to match to the  weight vector of the GT.

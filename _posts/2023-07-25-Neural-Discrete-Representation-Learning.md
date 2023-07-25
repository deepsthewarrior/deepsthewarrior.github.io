#### Posterior Collapse:
Posterior collapse occurs when the encoder fails to capture meaningful information about the input data in the latent variables. This can happen when the VAE is trained with very strong regularization or when the encoder and decoder networks are too powerful, making it easier for the VAE to rely solely on the decoder to reconstruct the data rather than learning useful representations in the latent space.

In cases of posterior collapse, the latent space distribution becomes very close to the prior distribution, effectively making the encoder's output almost independent of the input data. As a result, the latent variables lose their ability to encode useful information, and the VAE struggles to generate diverse and meaningful samples during the generative process.
Such a phenomenon is observed while using PixelCNN which is powerful decoder.

#### Discrete Representation:
Results of this paper show that using discrete representations performs as nice as the continuous representation models(VAE). Here the discrete representation vector for each point is called codes. And the codes are learnable vectors of fixed dimensions.
<figure style="text-align:center;">
  <img src="/images/NNR_arch.PNG" alt="Architecture" />
  <p class="img-caption"> e1 corresponds to the codebook features(Source: <a href="(https://www.baeldung.com/cs/cnn-receptive-field-size)">https://arxiv.org/abs/1711.00937v2/</a>)</p>
</figure>

#### Latent embedding space:
A latent embedding space is defined as e ∈ RK×D,(embedding vectors are called codes). D is the dimensionality and K indicates that it is K way categorical. An image x is fed to an encoder E producing ze(x). Then the authors use nearest negihbour look up to the shared  embedding space. Then we use the corresponding 1 one of K embedding vectors as the input to the Decoder.  

<figure style="text-align:center;">
  <img src="/images/NNR_equation_1.PNG" alt="argmin regime" />
  <p class="img-caption"> Selection  corresponding features from embedding space(Source: <a href="(https://www.baeldung.com/cs/cnn-receptive-field-size)">https://arxiv.org/abs/1711.00937v2/</a>)</p>
</figure>

We also learn a decoder G which takes in the corresponding learned codes for each latent vector and reconstructs the image. 
Usually the loss for any VAE is the sum of reconstruction loss and KL Divergence between the sampled z given the input x. However, here in our case, the method to choose the z for given input x q(z = k|x)  is deterministic and a uniform prior over z is implicitly imposed. Thus the KL divergence formula deduces to K which is constant for all cases. Thus, we can ignore the KL divergence term. 

#### Backprop:
The forward pass is pretty straightforward, with slight modification in choosing the input to the decoder. We introduce learnable codes as additional parameter and a deterministic method to map each ouptut from the encoder to the learned codes. 

For backprop, we need to pass the gradients from the input of the decoder to the output of the decoder. However, with the equation in the figure 1, its evident that, we cannot reach the encoder via backprop. We will end up optimizing the params from the codebook. Since, both Encoder output and decoder input are D dimensional, we just copy the gradients from the decoder's input to the encder's output. This will efficiently let the encoder to generate output such that reconstruction loss is reduced.

#### Loss:
Loss is composed of 
- Reconstruction Error to enforce the encoder and decoder learnings.  
- VQ - Vector Quantisation Objective, to move the embedding vectors(codes)
to the encoder outputs. Sg(x) stands for stopped gradient for x, where the gradients are detached for x and other vector is moved in space towards x. (Here x is the latent representation)
- Third term is the vice versa of the second loss term.


<figure style="text-align:center;">
  <img src="/images/NNR_loss.PNG" alt="Loss Equation" />
  <p class="img-caption"> (Source: <a href="(https://www.baeldung.com/cs/cnn-receptive-field-size)">https://arxiv.org/abs/1711.00937v2/</a>)</p>
</figure>

Encoder's outputs take part in all 3 terms while decoder's  output corresponds to first term only.

#### AutoRegressor:
After training, we use an autoregressor like PixelCNN for Images and WaveNet for audio. AutoRegressor generates output based on the learnt z and the autoregressive prior.


#### References:
1. https://arxiv.org/abs/1711.00937v2
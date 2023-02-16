---
layout: post
title: Unreliability of explanations in few shot prompting for textual reasoning
---

This blog is a report on the Learning from Limited Data in NLP. This blog summarizes the methods & results of the paper: <https://arxiv.org/pdf/2205.03401.pdf>. The paper summarizes how explanations can improve the performance of few shot learning in NLP on very large language models and it focuses on two textual reasoning tasks namely, Question Answering(QA) and Natural Language Inference(NLI).

We mainly discuss about the various experiments, models and datasets used, and the results generated. We will also discuss how the authors calibrated the models in order to understand the usage of these explanations post hoc.

In the first senction, I will give a short introduction of the paper and the motivation. In the second section, I will explain about the datasets used in the paper along with example prompts with explanations. I will then talk about the quality of generated explanations along with the accuracy results. The authors calibrate the models to get better results which will be discussed in the final section. I will end this blog with a short conclusion about future work and other possibilites.

## Introduction

Models like GPT-3, OPT and Instruct GPT give their best performance by learning from very few examples in-context. One can let the model explain itself in order to obtain explanations which can be useful to understands how the models work. Textual reasoning tasks like QA and NLP are used enormously for modelling LLMs with few training examples. But does prompting a LLM with explanations improve the overall performance of the model? If yes/no, what is the quality of the model generated explanations itself? Can we judge the quality of these explanations by just one performance metric?

We have large language models currently such as GPT-3(davinci), OPT, InstructGPT(text-davinci-001) and text-danvinci-002 which are known to have an effective performance for textual reasoning tasks. 

<figure style="text-align:center;">
  <img src="/images/intro.PNG" alt="Explanation" />
  <p class="img-caption">Prompting GPT-3 with an explanation</p>
</figure>

In the above example prompt with explanation, we see that the GPT-3 model generates a non factual explanation. The authors prove that these nonfactual explanations can help to calibrate the model. 
## What is an explanation?

As discussed above, an explanation is added to every input prompt to improve the performance of the model. Explanation is a summarized in-context view of the input prompt. But how does this look? And what are the different ways one can add an explanation to the prompt?

To answer these questions, we must understand the type of datasets used for this experiments and also their prompt structure.
The authors in the paper used three different datasets SYNTH, ADV HOTPOT and E-SNLI for the four models mentioned in the above section. The question answering task is performed with SYNTH and ADV HOTPOT, and the NLI task is performed with the E-SNLI dataset.

In Question Answering tasks, usually the prompt consists of bridge statements. These can be both supporting and distractor statements.

<figure style="text-align:center;">
  <img src="/images/data_prompt.PNG" alt="Prompt" />
  <p class="img-caption">Example prompt for SYNTH</p>
</figure>

The above mentioned prompt examples belongs to SYNTH dataset. SYNTH is a synthetic multi-hop QA dataset which consists of bridge questions with two supporting statements and distractor statements. In the above example, only *Mary hangs out Danielle* and *Danielle is a student* are the two supporting statements paired with many distractor statements. SYNTH always has the prompt structure as "B is profession, A is something of B"


<figure style="text-align:center;">
  <img src="/images/data_prompt3.PNG" alt="Prompt" />
  <p class="img-caption">Example prompt for ADV HOTPOT</p>
</figure>

The above mentioned prompt belongs to ADV HOTPOT dataset. ADV HOTPOT is a adversarial version of the English-language HOTPOT QA dataset and this augmented version is used because InstructGPT gives best performance on the adversarial setting of the dataset. As you can see, the prompt consists of two ground truth supporting statemenst and two adversarial statements.

In NLI tasks, the explanations are usually judged as entailed by/ neutral/ contradicted by based on the given explanations. E-SNLI is an English-language NLI classification dataset. In the below mentioned example prompt, there is a premise and hypothesis and the prediction label is neither with an explanation.

<figure style="text-align:center;">
  <img src="/images/data_prompt2.PNG" alt="Prompt" />
  <p class="img-caption">Example prompt for E-SNLI</p>
</figure>

### Explanations can be added to prompts in two different ways:

Standard few-shot in-context learning is using the input prompts without explanation. Explanations can be added to the prompt using two of the following methods:

#### Explain-then-predict:

In this method, the models generate an explanation first and then the prediction is followed. So, the generated label has influence of the generated explanation.

<figure style="text-align:center;">
  <img src="/images/ep.PNG" alt="Prompt" />
  <p class="img-caption">Example for Explain-then-Predict</p>
</figure>

In the prompt given above we can see that the explanation *Because Danielle is a student and Mary hangs out with Danielle* is generated first and then the label is given as *Mary*.

#### Predict-then-explain:

In this method, the label is generated first and then the explanation is followed. Hence, the generated explanation doesn't have any influence on the output label but the prompt explanation still has an influence on the generated label.

<figure style="text-align:center;">
  <img src="/images/pe.PNG" alt="Prompt" />
  <p class="img-caption">Example for Predict-then-Explain</p>
</figure>

Here we see that the explanation *because not every person is a girl* is generated after the label has been predicted.

### Did explanations improve few shot learning? 

Now that we know about E-P and P-E, lets analyze the performance of the four LLMs when explanations are given. The setup for this experiment the authors mentioned they have used maximum allowed shots for OPT and GPT-3 i.e., 16 for SYNTH, 6 for ADV HOTPOT and 32 for E-SNLI. The primary LM for the entire experiment setup is InstructGPT since it was the most effective model available during the experiment and hence, they used 5 groups of training shots and 3 groups for the rest three models.
The table below summarizes the results for all the three datasets: 

<figure style="text-align:center;">
  <img src="/images/exp1.PNG" alt="Prompt" />
  <p class="img-caption">Example for nonfactual explanation</p>
</figure>

We see that the performance has very small to moderate change on three models except for text-danvinci-002 which achieved better performance on E-P than few shot prompting. On InstructGPT, both QA datasets improve the performance on E-P and for NLI task, E-SNLI improves the performance on P-E. We can also observe that the performance of P-E over all three datasets is very inconsistent.

## Can LLMs generate factual and consistent explanation?
The explanations generated by the models can be measured in two different metrics:

### Factuality:

A factual explanation is always grounded within the input. It truthfully explains what the input context tends to convey to the model. It does not contain any distractor statements about the input context.

<figure style="text-align:center;">
  <img src="/images/factual.PNG" alt="Prompt" />
  <p class="img-caption">Example for nonfactual explanation</p>
</figure>

### Consistency: 

The authors followed the definition of a consistent expln(Alon Jacovi and Yoav Goldberg. 2021. Aligning faithful interpretations with their social attribution. Transactions of the Association for Computational Linguistics (TACL), 9:294–310) where a consistent expln always follows/entails the prediction exactly how humans perceive.

<figure style="text-align:center;">
  <img src="/images/consistent.PNG" alt="Prompt" />
  <p class="img-caption">Example for consistent explanation</p>
</figure>


### What is the quality of these generated explanations? 

In this section, we will discuss how accurate and consistent the generated explanations. As you can see the table below, only consistency is shown for E-SNLI because checking the factuality of an explanation would require additional knowledge which can't be added to the labels easily. Hence, we have a disconnect between the model and the reasoning in the explanations for E-SNLI. The authors perform the experiment on InstructGPT across three datasets but only on SYNTH for the other three models.

<figure style="text-align:center;">
  <img src="/images/accuracy.PNG" alt="Result" />
  <p class="img-caption">Quality of generated explanations</p>
</figure>

LLMs tend to generate consistent explanation but they are highly likely to be inconsistent. This is a major problem since it can deceive the user into believing something wrong.

The authors also check the reliabilty of explanations and the prediction accuracy. For SYNTH dataset, a non-reliable explanation usually means an incorrect prediction across all four LLMs. For ADV HOTPOT, factuality for InstructGPT's performance was around 80.0%. Now we understand that LLMs kind of hallucinate explanations but this gives a chance to spot where the LLM's reasoning is failing. In order to discover the connection between the reliability of an explanation and the accuracy of its prediction, we need to calibrate the models.

## Calibrating In-context learning using explanations:

From the above tabular results we see that its possible to automate the process of using the factuality of an explanation to generate the corresponding prediction. But can we automate this process?

The authors of this paper use InstructGPT as their base model for performance and calibration. This is because, during the time of this experiment InstructGPT was on of the most powerful models which had a significant room for development. So, now in the calibration process the authors calibrate InstructGPT model on all three datatsets and achieve an effective overall performance. 

Calibration was very easy to achieve on SYNTH since it has a perfectly controlled setting in the dataset. For ADV HOTPOT and E-SNLI, authors have used lexical matching to approximate semantic matching to reflect factuality. 

Now lets see one-by-one for three datasets on how to calibrate.

### Improving SYNTH:
The authors claim to achieve very good results on SYNTH dataset by calibration. They also claim that achieving the results was very easy on this dataset. 
As we already saw that P-E performs well on SYNTH, the authors calibrate and provide the results only for this setting. 
As we know that the prompt structure for SYNTH dataset is *[B is profession, A verb B]* which can be easily split into two sentences. An explanation is factual if and only if each of the two sentences match one sentence in context. They have observed that the accuracy increased from 52.4 to 74.8%. No model could achieve this performance when trained even with 16 training examples

## Learning based calibration framework:
This framework is used for the ADV HOTPOT and E-SNLI datasets. Lets now see how it works: 

### Framework:
Unlike classical calibration methods(provide reference) which use affine transformation on the probabilities only.(add the equation)

The authors have improvised the classical method by adding an extra term 'v' which is a scalar value that describes the factuality of an explanation: p&#770;
 = softmax(W[p; v] + b) where p&#770; is the tuned probabilities, v is the factor that incorporates the factuality assessment of an explanation and p is the vector of predicted probailities and it is associated with the class label of NLI/ probability score of the predicted answer in QA task. 

### Approximating factuality: 
Authors have used lexical overlap to approximate the factuality of an explanation which worked well for all the tasks.

#### For ADV HOTPOT
We have seen that ADV HOTPOT dataset has two sentences in an explanation. For the two sentences E<sup>(1)</sup> and E<sup>(2)</sup>, an explanation &epsilon; is generated. For each explanation E<sup>(i)</sup>, there are (e<sub>1</sub>, e<sub>2</sub>,..e<sub>i</sub>) tokens. Similarly, for P=(P<sup>(1)</sup>, P<sup>(2)</sup>, P<sup>(3)</sup>, P<sup>(4)</sup>) are context paragraphs and for each context paragraph P<sup>(i)</sup>, we have (p<sub>1</sub>, p<sub>2</sub>,...p<sub>i</sub>) tokens. 

Now, we can calculate the factuality of an explanation E^(i) as:



Add the equations and explain the math little

#### For E-SNLI
As we have seen that E-SNLI has a premise and hypothesis situation, so it does not involve the factuality concept.
Describe the math equations and explain little

The more the explanation overlaps with the premise, the higher the factuality.

### Calibrating ADV HOTPOT:
Add reference for setup theory. Selective QA used .To check the quality of calibration, the authors use coverage-accuracy-curve plot. A better calibrated model must be able to pick the questions it can perform best on, which gives higher AUC.

As we saw earlier, E-P performs better than P-E on ADV HOTPOT. Training data size is 6,32 and 64 and the authors show the results averaged from 5 different trials of different datasets.

### Results:
<figure style="text-align:center;">
  <img src="/images/res_adv.PNG" alt="Result" />
  <p class="img-caption">Results for ADV HOTPOT after calibration</p>
</figure>

<figure style="text-align:center;">
  <img src="/images/auc.PNG" alt="Result" />
  <p class="img-caption">AUC scores for ADV HOTPOT</p>
</figure>

### Calibrating E-SNLI:
Training data sizes vary from 32 to 128.

## Results:
<figure style="text-align:center;">
  <img src="/images/res_esnli.PNG" alt="Result" />
  <p class="img-caption">Results for E-SNLI after calibration</p>
</figure>



## Conclusion:






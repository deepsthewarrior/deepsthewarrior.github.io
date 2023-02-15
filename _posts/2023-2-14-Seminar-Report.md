---
layout: post
title: Unreliability of explanations in few shot prompting for textual reasoning
---

This blog is a report on the Learning from Limited Data in NLP. This blog summarizes the methods & results of the paper: <https://arxiv.org/pdf/2205.03401.pdf>. The paper summarizes how explanations can improve the performance of few shot learning in NLP on very large language models and it focuses on two textual reasoning tasks namely, Question Answering(QA) and Natural Language Inference(NLI).

In this blog, I will discuss about the paper (reference link to the paper). We mainly discuss about the various experiments, models and datasets used, and the results generated. We will also discuss how the authors calibrated the models in order to understand the usage of these explanations post hoc.

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

In NLI tasks, the explanations are usually judged as entailed by/ neutral/ contradicted by based on the given explanations. E-SNLI 
<figure style="text-align:center;">
  <img src="/images/data_prompt2.PNG" alt="Prompt" />
  <p class="img-caption">Example prompt for E-SNLI</p>
</figure>

### Explanations can be added to prompts in two different ways:

#### Explain-then-predict:

In this method, the models generate an explanation first and then the prediction is followed. So, the generated label has influence of the generated explanation.
<figure style="text-align:center;">
  <img src="/images/ep.PNG" alt="Prompt" />
  <p class="img-caption">Explain-then-predict</p>
</figure>

#### Predict-then-explain:

In this method, the label is generated first and then the explanation is followed. Hence, the generated explanation doesn't have any influence on the output label but the prompt explanation still has an influence on the generated label.
<figure style="text-align:center;">
  <img src="/images/pe.PNG" alt="Prompt" />
  <p class="img-caption">Predict-then-explain</p>
</figure>

### Did explanations improve few shot learning? 

<figure style="text-align:center;">
  <img src="/images/exp1.PNG" alt="Prompt" />
  <p class="img-caption">Example for nonfactual explanation</p>
</figure>

## Can LLMs generate factual and consistent explanation?
The explanations generated by the models can be measured in two different metrics:

### Factuality:

A factual explanation is always grounded within the input. It truthfully explains what the input context tends to convey to the model. It does not contain any distractor statements about the input context.
(Add image for factual expln example)

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


#### What is the quality of these generated explanations? 
(Add accuracy results table)
<figure style="text-align:center;">
  <img src="/images/accuracy.PNG" alt="Result" />
  <p class="img-caption">Quality of generated explanations</p>
</figure>

## Calibrating models:

From the above tabular results we see that its possible to automate the process of using the factuality of an explanation to generate the corresponding prediction.

The authors of this paper use InstructGPT as their base model for performance and calibration. This is because, during the time of this experiment InstructGPT was on of the most powerful models which had a significant room for development. So, now in the calibration process the authors calibrate InstructGPT model on all three datatsets and achieve different set of performances,

Now lets see one-by-one for three datasets on how to calibrate.

### Calibrating SYNTH:
The authors claim to achieve very good results on SYNTH dataset by calibration. They also claim that achieving the results was very easy on this dataset. 
As we already saw that P-E performs well on SYNTH, the authors calibrate and provide the results only for this setting. 

They have observed that the accuracy increased from 52.4 to 74.8%
No model could achieve this performance when trained even with 16 training examples

## Learning based calibration framework:
### Framework:
Unlike classical calibration methods(provide reference) which use affine transformation on the probabilities only.(add the equation)

The authors have improvised the classical method by adding an extra term 'v' which is a scalar value that describes the factuality of an explanation
(add the new equation and any references) and P cap is the tuned probabilities. 


### Approximating factuality: 
Authors have used lexical overlap to approximate the factuality of an explanation. They have concluded that lexical overlap worked well for all the tasks.
#### For ADV HOTPOT
We have seen that ADV HOTPOT dataset has two sentences in an explanation(add an example). 
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


## Summary of the findings:


## Conclusion:






# EA_project_attention_flow

This repository aims to study how attention flow can be used to determine which input tokens are given the most "weight" when it comes to decision making. We study its application on sentiment analysis tasks, as well as image classification. 
We develop methods to evaluate how meaningful this information for classification. For that purpose, we study how carefully designed perturbations in the inputs affect the classification outcome. We also try to compare the performance of attention flow to other methods, notably *Attention rollout*.

## Attack methods on sst2 dataset using BERT

`BERT_attack_sst2.ipynb`: the notebook is detailed regarding its implementation

The SST-2 (Stanford Sentiment Treebank v2) is a dataset containing sentences and labels indicating the underlying sentiment. This part is based primarily on https://github.com/dataflowr/Project-attention_flow and uses some of its code to calculate attention flow on sentences from the dataset. The main idea is to implement an attack, in which we modify words that are associated with a high attention score and compare how the model (BERT) succeeds in classifying the image.

## Attention Flow on images using DeiT

`attention_flow.py`: class implementation
`attention_flow.ipynb`: example usage
`generate_masks.ipynb`: generate masks for `test_images/` and saves them in `heatmaps.csv`

We took inspiration from https://github.com/jacobgil/vit-explain (which implements **Attention Rollout** and **Attention Grad Rollout**) to implement **Attention Flow** on an image Transformer (DeiT model) ourselves.

## Attack methods on imagenette dataset using DeiT

`DeiT_attack_methods.ipynb`

We try to implement an attack method on the imagenette dataset using rollout to try to determine its effect. The main idea is to determine which parts of the image are the most important and modify their value. We compare this method to a baseline one which pixel values are modified randomly. We also implement our own **Attention Flow** implementation but on a much smaller dataset (9 images) since it takes longer to run. We still need to implement it on a much larger dataset to get more significant results.
# HTCV: The final advances to the Decomposer model

Guiding the decomposition learning process to reduce ambiguity.
After multiple runs of training the model with our naive loss function approach, we realized that without any ground truths for occlusion, shadow, and light masks, the self-supervised model struggled to distinguish between individual distortions. In many cases, the shadow mask dominated in extracting all distortions + background artifacts from the distorted image. Therefore, to guide the model in the right direction, we made the decision to create pseudo-labels for shadow, light, and occlusion masks. Finally, we split the training into multiple stages, where the goal is to pretrain each head for a specific task separately, based on the pseudo-labels.

## Table of contents
1. [Source Image Reconstruction ](#source-image-reconstruction) \
1.1 [Loss function](#loss-function)
2. [Light and Shadow Mask Reconstruction](#light-and-shadow-mask-reconstruction) \
2.1 [Generating Pseudo Labels](#generating-pseudo-labels) \
2.2 [Loss function](#loss-function)
3. [Occlusion Mask Reconstruction](#occlusion-mask-reconstruction) \
3.1 [Generating Pseudo Labels](#generating-pseudo-labels) \
3.2 [Loss function](#loss-function)
3.2.1 [Binary Mask](#binary-mask) \
3.2.2 [RGB Mask](#rgb-mask)

## Source Image Reconstruction
### Loss function

## Light and Shadow Mask Reconstruction
### Generating Pseudo Labels
### Loss function

## Occlusion Mask Reconstruction
### Generating Pseudo Labels

![Fig 3.1](figures/3-Advanced_Decomposition_Learning/Occ_input_and_target.png)
**Fig 3.1:** *Input images (top) and generated occlusion pseudo targets (bottom).*
### Loss function
Again two stages. First Only training to predict binary mask. Then training to predict Mask with RGB values.

#### Binary Mask
BCEWithLogitsLoss. ```True``` values for mask very underrepresented -> Positive Weight computed each batch: Ratio of ```False``` to ```True``` values in batch. \
```torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)```
#### RGB Mask
MAE of reconstruction and Input. 2 Versions - both with different regularizations.
1. Reconstruction = SL_Target XOR Occ_Mask * Occ_RGB
2. Reconstruction = (Source_Image_Target + light_pred) * shadow_pred XOR (Occ_Mask * Occ_RGB)

***Version 1*** without regularization:
> Insert image

The model learns to simply reconstruct the input image in its occlusion rgb prediction and the binary mask predicts only true values, i.e. the occlusion branch takes over the complete reconstruction like a Autoencoder. \

***Version 1*** with binary mask regularization:
> Insert image

The model again learns to somewhat reconstruct the original image, but the binary mask now also sticks to the noisy binary mask target. Since our targets are very poor, we want them only to be used as a guide and want the model to lern to generalize from there on. This might still be a simple hyperparameter tuning problem as the weighting of the binary mask subloss might be too high. \

***Version 1*** with mask decay regularization:
![Fig 1](figures/3-Advanced_Decomposition_Learning/Occ_pretraining_V1_w_md.png)

The model again learns to somewhat reconstruct the original image, but the mask decay term now tells the model to predict as little positives, ie., 1s, as possible. The goal is to regularize strong enough so that the model predicts almost no positives, only when there really is an occlusion. This might still be a simple hyperparameter tuning problem. \
The issue with this approach is that the underlying image with the generated shadows and light images (the sl pseudo targets) are not that accurate, so the occlusion mask really has to rely on it's own RGB input reconstruction. \
But since we have also pretrained and finetuned the gt and sl models to a point where the shadow and light are modeled better than the pseudo labels, we can use the gt and sl models to generate the underlying image instead of using the generated sl pseudo target for the occlusion modeling. \

***Version 2*** with mask decay regularization:

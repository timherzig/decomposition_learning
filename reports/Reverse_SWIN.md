# HTCV: Novel Reverse Video SWIN Transformer as the Decoder
We construct a reverse Video SWIN Transformer, which basically reverses the steps in the original Video SWIN Transformer, as an alternative to the UNet decoder. It takes the output from the encoder as input and passes it through 4 stages, which eventually restores the dimension of the latent representation from the encoder to the dimension of the original batches. 

Each of the first three stages consists of Video SWIN Tranformer blocks and a patch splitting layer. As a reversal of the patch merging layer in the original Video SWIN Transformer, the patch splitting layer doubles the number of features of each patch by a linear layer and and splits them into 4 patches. As a result, after each stage, the number of patches is four times the original and the number of features of each patch halved. 

Eventually, transposed convolution is applied to project the patches back to the desired dimensions of the masks. 

## Discussion
We only ran some preliminary experiments on this novel decoder and the results looked noisy and far from satisfatory. For future work, we hope to finetune the architecture and parameters more strategically, e.g. adding skip connections like the ones in the UNet model, reducing the number of layers in each stage to see if a lighter architecture can capture useful information more effectively, etc.
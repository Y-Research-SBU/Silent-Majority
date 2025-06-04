# [Uncovering Memorization Effect in the Presence of Spurious Correlations](http://arxiv.org/abs/2501.00961) (Nature Communications)

This is the official PyTorch implementation of our  [![arXiv](https://img.shields.io/badge/arXiv-2501.00961-b31b1b.svg)](https://arxiv.org/abs/2501.00961).

## Overview
In this work, we present the first systematic study on the role of different neurons in memorizing different group information, and confirm the existence of critical neurons where memorization of spurious correlations occurs.

We show that modifications to specific critical neurons can significantly affect model performance on the minority groups, while having almost negligible impact on the majority groups.

We propose spurious memorization as a new perspective on explaining the behavior of critical neurons in causing imbalanced group performance between majority and minority groups.

## Environment Preparation
The required packages are listed in `requirements.txt`. Make sure you having `.../Uncovering-Spurious-Memorization` as the working directory when running any scripts.

## Dataset Preparation

### Waterbirds
The Waterbirds dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).

Our code expects you set `.../waterbird_complete95_forest2water2` to the argument `--waterbird_dataset_path`.

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). 

### CelebA
Our code expects the `img_align_celeba` in the `celeba` directory:
```
.../celeba/img_align_celeba/
```
You need to set `.../celeba` to the argument `--celeba_dataset_path`.

You can download these dataset files from [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Model Checkpoints Preparation
Download the following checkpoints to `experiments/` folder, keep the name unchanged:
| Model      | Pretrained Checkpoint Download Link                                                                                        |
|------------|----------------------------------------------------------------------------------------------------------------------------|
| resnet50   | https://download.pytorch.org/models/resnet50-19c8e357.pth                                                                  |
| deit_tiny  | https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth                                                     |
| deit_small | https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth                                                    |
| deit_base  | https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth                                                     |
| deit_large | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth |

We finetuned the pretrained models for 40 and 20 epochs via ERM on Waterbirds and CelebA respectively, using `runSpuriousCorrelationPretrainingErm.sh`.

## Part I: Identifying the Existence of Critical Neurons
In this section, we validate the existence of critical neurons in the presence of spurious correlations.

| Bash Scripts                                         | Objectives                                                                                          |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| runSpuriousCorrelationPretrainingErm.sh              | Train the model on celeba/waterbirds via ERM.                                                       |
| runStudyChannelwiseLargestNeuronExperiments.sh       | Investigate the impact of modifying the top-k largest neurons in a model trained using ERM.         |
| runStudyChannelwiseMostActivatedNeuronExperiments.sh | Investigate the impact of modifying the top-k most activated neurons in a model trained using ERM.  |
| runStudyLargestNeuronByThresholdExperiments.sh       | Investigate the impact of modifying the top-x percent largest neurons in a model trained using ERM. |

To reproduce the Figure 1/2/3/4/5/11/12 and Table 11/13/14, you can run
```
python scripts/StudyChannelwiseMostActivatedNeuronExperiments.py --experiment_dataset="waterbirds" \
                                                                    --experiment_split="train" \
                                                                    --celeba_dataset_path=".../celeba" \
                                                                    --waterbird_dataset_path=".../waterbird_complete95_forest2water2" \
                                                                    --modification_mode="zero" \
                                                                    --noise_std=0 \
                                                                    --row_back_n_epochs=0 \
                                                                    --top_k=3
```
or 
```
python scripts/StudyChannelwiseLargestNeuronExperiments.py --experiment_dataset="waterbirds" \
                                                            --experiment_split="train" \
                                                            --celeba_dataset_path=".../celeba" \
                                                            --waterbird_dataset_path=".../waterbird_complete95_forest2water2" \
                                                            --modification_mode="zero" \
                                                            --noise_std=0 \
                                                            --row_back_n_epochs=0 \
                                                            --top_k=3
```
by setting the argument accordingly.

If you are performing random initialization/noise related experiment, you have to average the result from 10 experiments documented in the output txt. For example if the output text reads as
```
1.0000, 0.9891, 0.8929, 0.9858
modified neurons name, index and value
conv1.weight, 28, 4.643474578857422
0.9987, 0.8208, 0.6371, 0.9315
0.9956, 0.8102, 0.6371, 0.9299
0.9969, 0.8111, 0.6604, 0.9299
0.9973, 0.8067, 0.6449, 0.9315
0.9965, 0.8018, 0.6589, 0.9408
0.9973, 0.8129, 0.6402, 0.9268
0.9969, 0.8137, 0.6262, 0.9252
0.9987, 0.8151, 0.6324, 0.9283
0.9965, 0.8049, 0.6121, 0.9221
0.9987, 0.8262, 0.6386, 0.9283
```
Then we calculate the average by column of the last 10 rows (each row denotes the group 0-3 accuracy with random initialization/noise) and subtract it from the first row data (the original group 0-3 accuracy without pruning) to get the final result.

## Part II: Spurious Memorization by Critical Neurons
In this section, we take a further step in investigate the cause of imbalanced group performance under spurious correlation, particularly focusing on the discrepancy in the test accuracy between majority and minority groups. 

The script `runSpuriousCorrelationFinetuningPruning.sh` aims to finetune the model with critical neuron pruned, in order to closing the gap between the majority and minority group accuracy.

To reproduce the Table 4-10, you can run `runSpuriousCorrelationFinetuningPruning.sh` by setting arguments accordingly.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{you2025silent,
  title={The Silent Majority: Demystifying Memorization Effect in the Presence of Spurious Correlations},
  author={You, Chenyu and Dai, Haocheng and Min, Yifei and Sekhon, Jasjeet S and Joshi, Sarang and Duncan, James S},
  journal={arXiv preprint arXiv:2501.00961},
  year={2025}
}
```

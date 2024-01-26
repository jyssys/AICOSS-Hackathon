# 2023 AICOSS 위성 이미지 다중 분류 해커톤
![banner](https://github.com/jyssys/AICOSS-Hackathon/assets/22981960/e1fd58da-c143-442b-8367-fbb365c5f0a1)
This is the 2023 AICOSS Hackathon '무적환공' Team co-hosted by University of Seoul and Hyundai XiteSolution.

Team information
- 김창현(Chang-Hyun Kim)
  - B.S in University of Seoul, Dept of Environmental Engineering and Big Data Analysis.
  - M.S in University of Seoul, Dept of Statistics Data Science
- 정의수(Eui-Soo Jung) [author]
  - B.S in University of Seoul, Dept of Environmental Engineering and Big Data Analysis.

## Update

01/24/2024 : we have uploaded the code

## Informations
- OS : ubuntu 20.04
- python : 3.8
- CUDA : 11.4
- NVIDIA Driver version : 470,82,01
- GPU : NVIDIA Geforce rtx3090 (24GB)
- Random Seed : 605

## Requirements
**We didn't make requirements.txt, so I attached a separate version.**

- pytorch : 1.13.1+cu117
- torchvision : 0.14.1+cu117
- Other libraries(numpy, pandas, timm etc.) are up to date.

## Dataset Preparation
**Download the satellite image dataset from [here]**
(https://dacon.io/competitions/official/236201/data)

The dataset folder should have the following below structure:
<br>

     └── data
         |
         ├── test (folder)
         ├── train (folder)
         ├── sample_submission.csv
         ├── test.csv
         └── train.csv

- There are test images (43,665) in the data/test folder.
- There are train images (65,496) in the data/train folder.

## Training
All you need to do is run main.py.

<br>

     python3 main.py

- Finally, after learning is completed, 7 csv will be created to match the config in the results folder. 
  [updated_submission_{version+1}.csv]
- After that, a csv with 7 csv soft voted(ensembled) will be created in the folder. 
  [ensemble_results.csv]
- 7 models of weight files are created in the weights folder. 
  [model_config_{version+1}.pth]

<br/>
**'augmentation' folder and 'multi_augmentation.py' were only used for experiments and not in the learning process!**
- augmentation folder : This is a baseline code for data synthesis through GAN and active learning in next research.
- multi_augmentation.py : This is augmentation by crop with small kernel(crop_size : 94, stride : 47)

## Results
<img width="992" alt="image" src="https://github.com/jyssys/AICOSS-Hackathon/assets/22981960/ca90a538-76f0-46b7-ab70-b4f0506fa06e">
<img width="689" alt="image" src="https://github.com/jyssys/AICOSS-Hackathon/assets/22981960/7278e353-d403-4fd5-bdfa-6691c76b66f1"> <br/>
(Loss : BCELogitLoss / Evaluation Score : mAP) <br/>


## Competition Results
- Public : 2nd
- Private : 5th
- Final : 3rd (서울시립대학교 공과대학장상 2등상)

## Reference
[1] TAN, Mingxing; LE, Quoc. Efficientnet: Rethinking model scaling for convolutional neural networks. In: International conference on machine learning. PMLR, 2019. p. 6105-6114. <br/>
[2] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017. <br/>
[3] Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020). <br/>
[4] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017). <br/>

## Presentation Materials
We would appreciate it if you could refer to this pdf.
[2023 AICOSS 무적환공.pdf](https://github.com/jyssys/AICOSS-Hackathon/files/14035024/2023.AICOSS.pdf)



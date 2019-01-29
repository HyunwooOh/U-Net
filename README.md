# U-Net
- Tensorflow implementations of U-Net: Convolutional Networks for Biomedical Image Segmentation.
- U-Net 을 구현했습니다.
- <img src= "/assets/network.png" width="100%" height="100%">

## Requirements
- Python v3.6
- tensorflow v1.4

## How to use
- `python3 main.py --todo [train or test or img_aug]`

### Data Source
- Original data set can be downloaded from [isbi challenge cite](http://brainiac2.mit.edu/isbi_challenge/).
- [Image Augmentation](img_aug.py) : Original data set 이 적어서 rotation, cropping 을 이용해 데이터를 생성했습니다.
    - [Original data](./original_data) (30 * (512, 512)) - rotation - central cropping - random cropping -> [new data](./new_data) (300 * (256, 256))

### 과적합 방지
- 데이터셋 300 -> 1500 : 정확도 90% -> 92%
- 배치정규화 : 정확도 92% -> 93%

### Result
- 1000 epoch 학습한 후,
    - train data 로 한 결과
        - <img src= "/assets/train_data_result.png" width="100%" height="100%">
    - test data 로 한 결과
        - <img src= "/assets/test_data_result.png" width="100%" height="100%">
### To be added
- ~~Image Augmentation~~
- ~~Batch Normalization~~
- Drop Out


## References
- Studies
    - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Github repositories
    - https://github.com/zhixuhao/unet
    
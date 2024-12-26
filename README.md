# A Bearing Fault Diagnosis Framework Based on Few-Shot Learning with Distribution Consistency and Structural Reparameterization

This is our implemented source code for the paper "A Bearing Fault Diagnosis Framework Based on Few-Shot Learning with Distribution Consistency and Structural Reparameterization" accepted by International Conference on Green Technology and Sustainable Development 7th
## Environment
```bash 
conda create -n BEARING python=3.10.12 -y
conda activate BEARING
pip install -r requirements.txt
```

## Dataset
[CWRU Download Link](https://engineering.case.edu/bearingdatacenter)


## Getting Started
- Installation
``` bash
git clone https://github.com/giabao804/few-shot-structural-rep.git
```
```bash
cd few-shot-structural-rep
```
- Training for 1 shot
``` bash
python3 train_1shot.py --dataset 'CWRU' --training_samples_CWRU 30 --model_name 'few-shot-structural'
```
- Testing for 1 shot
```bash
python3 test_1shot.py --dataset 'CWRU' --best_weight 'PATH TO BEST WEIGHT'
```
- Training for 5 shot
``` bash
python3 train_5shot.py --dataset 'CWRU' --training_samples_CWRU 60
```
- Testing for 5 shot
```bash
python3 test_5shot.py --dataset 'CWRU' --best_weight 'PATH TO BEST WEIGHT'
```

## Contact
Please feel free to contact me via email bao.tg212698@sis.hust.edu.vn or giabaotruong.work@gmail.com if you need anything related to this repo!
## Citation
If you feel this code is useful, please give us 1 ⭐ and cite our paper.
```bash
@inproceedings{truong2024bearing,
  title={A Bearing Fault Diagnosis Framework Based on Few-Shot Learning with Distribution Consistency and Structural Reparameterization},
  author={Truong, Gia-Bao and Than, Nhu-Linh and Vu, Manh-Hung and Pham, Van-Truong and Nguyen, Thi-Hue and Tran, Thi-Thao},
  booktitle={International Conference on Green Technology and Sustainable Development},
  pages={162--172},
  year={2024},
  organization={Springer}
}

```





# GenDexGrasp
Code Repository for ICRA 2023 Submission paper GenDexGrasp: Generalizable Dexterous Grasping

by [Puhao Li](https://github.com/Xiaoyao-Li)<sup> *</sup>, [Tengyu Liu](http://tengyu.ai/)<sup> *</sup>, [Yuyang Li](https://github.com/YuyangLee), [Yiran Geng](https://github.com/GengYiran), [Yixin Zhu](https://yzhu.io/), [Yaodong Yang](https://www.yangyaodong.com/), [Siyuan Huang](https://siyuanhuang.com/)
![Teaser](./assets/figures/teaser.png)

+ [ ] illustration here...





## Pipeline

+ [ ] We propose ...

![pipelinde](assets/figures/pipeline.png)



## Dependencies

Run the following to install a subset of necessary python packages for our code

```sh
pip install -r requirements.txt
```

Note that the `pytorch_kinematics` dependency is modified, you should install it from the source code in `thirdparty/pytorch_kinematics/`



## Data Preparation

#### Robots and Objects

We train and test on 58 daily objects from the YCB and ContactDB dataset, together with 5 robotic hands(EZGripper, Barrett Hand, Robotiq-3F, Allegro and Shadowhand) ranging from two to five fingers.

#### Grasp Dataset



#### Contact Map Dataset



## Usage

Train the 


## TODO List

+ [x] Inference
  - [x] Model Checkpoints
  - [x] Inference Code
+ [ ] Train
  - [ ] Training Data (URL)
  - [x] Training Code
+ [ ] Env
  - [x] Dependency (requirements.txt)
  - [x] pytorch_kinematics
  - [ ] IsaacGym

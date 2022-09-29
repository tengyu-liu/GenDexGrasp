# GenDexGrasp: Generalizable Dexterous Grasping
Code Repository for ICRA 2023 Submission paper GenDexGrasp: Generalizable Dexterous Grasping

by [Puhao Li](https://github.com/Xiaoyao-Li)<sup> *</sup>, [Tengyu Liu](http://tengyu.ai/)<sup> *</sup>, [Yuyang Li](https://github.com/YuyangLee), [Yiran Geng](https://github.com/GengYiran), [Yixin Zhu](https://yzhu.io/), [Yaodong Yang](https://www.yangyaodong.com/), [Siyuan Huang](https://siyuanhuang.com/)
![Teaser](./assets/figures/teaser.png)

+ [ ] arxiv link
+ [ ] project web link

## Pipeline

+ [ ] illustration here...

![pipelinde](assets/figures/pipeline.png)

## Dependencies

Run the following to install a subset of necessary python packages for our code

```sh
pip install -r requirements.txt
```

Note that the `pytorch_kinematics` dependency is modified, you should install it from the source code in `thirdparty/pytorch_kinematics/`

## Data Preparation

#### Robots and Objects

We train and test on 58 daily objects from the [YCB](https://www.ycbbenchmarks.com/) and [ContactDB](https://contactdb.cc.gatech.edu/) dataset, together with 5 robotic hands(EZGripper, Barrett Hand, Robotiq-3F, Allegro and Shadowhand) ranging from two to five fingers.

You can download the `data.zip` from [Google Drive](https://drive.google.com/file/d/1WRV7m9AAfDOFE6Z9InIRJwhhlRUSQzCX/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1UGC9Nhqp0C799bJ7IXz2iQ?pwd=md8u), then extract it to the root as

```sh
GenDexGrasp
+-- data
|  +-- object
|  |  +-- contactdb
|  |  +-- ycb
|  +-- urdf
|  |  ...
```

#### Grasp Dataset



#### Contact Map Dataset



#### IsaacGym Assets

We create a testing task using IsaacGym simulators to evaluate the stability of our generated grasp pose for objects and robotic hands. You can download the `env.zip` from [Google Drive](https://drive.google.com/file/d/1M_biyC7XcajSvat9FENI93kQtMA46h_3/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1y1Rg8GyfZ2mIZZbscDoEqg?pwd=q8n1), and extract it to the root as same as `data.zip` to build the tasks and assets.

## Usage

Train the 


## TODO

+ [x] Inference
  - [x] Model Checkpoints
  - [x] Inference Code
+ [ ] Train
  + [ ] Grasping Data with DFC
  + [x] Training Data (URL)
  + [x] Training Code
+ [ ] Env
  - [x] Dependency (requirements.txt)
  - [x] pytorch_kinematics
  - [ ] IsaacGym

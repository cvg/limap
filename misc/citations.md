# Citations for the supported methods

LIMAP builds on a number of external line detectors, descriptors, matchers,
vanishing point estimators and plane detectors. Each is the work of its own
authors, and the BibTeX entry below is the one they ask for. Please cite the
methods you use.

## Line detectors

### LSD

[pytlsd](https://github.com/iago-suarez/pytlsd) — Python bindings by Iago Suárez
to the original LSD implementation.

```bibtex
@article{von2008lsd,
  title={{LSD}: A fast line segment detector with a false detection control},
  author={Von Gioi, Rafael Grompone and Jakubowicz, Jeremie and Morel, Jean-Michel and Randall, Gregory},
  journal={IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI)},
  volume={32},
  number={4},
  pages={722--732},
  year={2008},
  publisher={IEEE}
}
```

### SOLD2

[SOLD2](https://github.com/cvg/SOLD2) — also used as a line descriptor and
matcher.

```bibtex
@inproceedings{pautrat2021sold2,
  title={{SOLD2}: {Self-Supervised} {Occlusion-Aware} Line Description and Detection},
  author={Pautrat, R{\'e}mi and Lin, Juan-Ting and Larsson, Viktor and Oswald, Martin R and Pollefeys, Marc},
  booktitle={Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

### HAWP

[HAWP](https://github.com/cherubicXN/hawp). LIMAP uses **HAWPv3**, the
self-supervised model of the journal version; please cite both.

```bibtex
@article{xue2023holistically,
  title={Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning},
  author={Xue, Nan and Wu, Tianfu and Bai, Song and Wang, Fu-Dong and Xia, Gui-Song and Zhang, Liangpei and Torr, Philip H. S.},
  journal={IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI)},
  year={2023}
}

@inproceedings{xue2020holistically,
  title={Holistically-attracted wireframe parsing},
  author={Xue, Nan and Wu, Tianfu and Bai, Song and Wang, Fudong and Xia, Gui-Song and Zhang, Liangpei and Torr, Philip HS},
  booktitle={Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

### TP-LSD

[TP-LSD](https://github.com/Siyuada7/TP-LSD)

```bibtex
@inproceedings{huang2020tp,
  title={{TP-LSD}: Tri-points based line segment detector},
  author={Huang, Siyu and Qin, Fangbo and Xiong, Pengfei and Ding, Ning and He, Yijia and Liu, Xiao},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2020}
}
```

### DeepLSD

[DeepLSD](https://github.com/cvg/DeepLSD) — the default line detector in LIMAP.

```bibtex
@InProceedings{Pautrat_2023_DeepLSD,
    author = {Pautrat, Rémi and Barath, Daniel and Larsson, Viktor and Oswald, Martin R. and Pollefeys, Marc},
    title = {{DeepLSD}: Line Segment Detection and Refinement with Deep Image Gradients},
    booktitle = {Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
}
```

## Line descriptors and matchers

### LBD

[pytlbd](https://github.com/iago-suarez/pytlbd) — Python bindings by Iago Suárez
to the LBD descriptor and its line matching.

```bibtex
@article{zhang2013efficient,
  title={An efficient and robust line segment matching approach based on {LBD} descriptor and pairwise geometric consistency},
  author={Zhang, Lilian and Koch, Reinhard},
  journal={Journal of Visual Communication and Image Representation},
  volume={24},
  number={7},
  pages={794--805},
  year={2013},
  publisher={Elsevier}
}
```

### LineTR

[LineTR](https://github.com/yosungho/LineTR)

```bibtex
@article{yoon2021line,
  title={Line as a Visual Sentence: {Context-Aware} Line Descriptor for Visual Localization},
  author={Yoon, Sungho and Kim, Ayoung},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  volume={6},
  number={4},
  pages={8726--8733},
  year={2021},
  publisher={IEEE}
}
```

### L2D2

[L2D2](https://github.com/hichem-abdellali/L2D2)

```bibtex
@inproceedings{abdellali2021l2d2,
  title={{L2D2}: Learnable Line Detector and Descriptor},
  author={Abdellali, Hichem and Frohlich, Robert and Vilagos, Viktor and Kato, Zoltan},
  booktitle={Proc. of the International Conf. on 3D Vision (3DV)},
  year={2021}
}
```

### SuperPoint

[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) — used for
the endpoint-based line matchers.

```bibtex
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR) Workshop on Deep Learning for Visual SLAM},
  year={2018}
}
```

### SuperGlue

[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) — used
together with [SuperPoint](#superpoint) for endpoint matching, so please cite
both.

```bibtex
@inproceedings{sarlin2020superglue,
  title={{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  author={Sarlin, Paul-Edouard and DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

### GlueStick

[GlueStick](https://github.com/cvg/GlueStick) — the default line matcher in
LIMAP, and its joint point-line matcher
(`cfgs/structure_triangulation/gluestick_joint.yaml`).

```bibtex
@inproceedings{pautrat2023gluestick,
  title={Gluestick: Robust image matching by sticking points and lines together},
  author={Pautrat, R{\'e}mi and Su{\'a}rez, Iago and Yu, Yifan and Pollefeys, Marc and Larsson, Viktor},
  booktitle={Proc. of the International Conf. on Computer Vision (ICCV)},
  year={2023}
}
```

### RoMa

[RoMa](https://github.com/Parskatt/RoMa) — the dense matcher behind the custom
LIMAP line matcher, also usable on its own for dense point matching.

```bibtex
@inproceedings{edstedt2024roma,
  title={{RoMa}: Robust Dense Feature Matching},
  author={Edstedt, Johan and Sun, Qiyu and B{\"o}kman, Georg and Wadenb{\"a}ck, M{\aa}rten and Felsberg, Michael},
  booktitle={Proc. of the Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Vanishing point estimators

### JLinkage

[JLinkage](https://github.com/B1ueber2y/JLinkage) — the default vanishing point
estimator in LIMAP.

```bibtex
@inproceedings{toldo2008robust,
  title={Robust multiple structures estimation with j-linkage},
  author={Toldo, Roberto and Fusiello, Andrea},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2008}
}
```

### Progressive-X

[Progressive-X](https://github.com/danini/progressive-x)

```bibtex
@inproceedings{barath2019progressive,
  title={Progressive-{X}: Efficient, anytime, multi-model fitting algorithm},
  author={Barath, Daniel and Matas, Jiri},
  booktitle={Proc. of the International Conf. on Computer Vision (ICCV)},
  year={2019}
}
```

## Plane detectors

### PxwPlanar

[PixelwisePlanarity](https://github.com/alpayozkan/PixelwisePlanarity) — the
default plane detector in LIMAP. It runs on a
[MoGe-2](https://github.com/microsoft/MoGe) backbone, so please cite both.

```bibtex
@inproceedings{yavuz2026pixel,
  title={{Pixel-wise Planarity for High-Precision Monocular Plane Segmentation}},
  author={Yavuz, Ahmetcan and Ozkan, Alpay and Pautrat, R{\'e}mi and Liu, Shaohui and Pollefeys, Marc},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2026}
}

@inproceedings{wang2025moge2,
  title={{MoGe-2}: Accurate Monocular Geometry with Metric Scale and Sharp Details},
  author={Wang, Ruicheng and Xu, Sicheng and Dong, Yue and Deng, Yu and Xiang, Jianfeng and Lv, Zelong and Sun, Guangzhong and Tong, Xin and Yang, Jiaolong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

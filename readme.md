# [CVPR 2023 Highlight] Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation
[Paper](https://arxiv.org/abs/2303.12246) | [Short Presentation](https://youtu.be/NWUf4hd571E) | [Long Presentation](https://youtu.be/JPvoObEYCAo)

## Motivation
Endow any estimated pose with **provably correct** performance guarantees, i.e., **a worst-case error bound** from the groundtruth pose

## Abstract
The two-stage object pose estimation paradigm first detects semantic keypoints on the image and then estimates the 6D pose by minimizing reprojection errors. Despite performing well on standard benchmarks, existing techniques offer no provable guarantees on the quality and uncertainty of the estimation. In this paper, we inject two fundamental changes, namely **conformal keypoint detection** and **geometric uncertainty propagation**, into the two-stage paradigm and propose the first pose estimator that endows an estimation with provable and computable worst-case error bounds. On one hand, conformal keypoint detection applies the statistical machinery of _inductive conformal prediction_ to convert heuristic keypoint detections into circular or elliptical prediction sets that cover the groundtruth keypoints with a user-specified marginal probability (e.g., 90%). Geometric uncertainty propagation, on the other, propagates the geometric constraints on the keypoints to the 6D object pose, leading to a **Pose UnceRtainty SEt (PURSE)** that guarantees coverage of the groundtruth pose with the same probability. The PURSE, however, is a nonconvex set that does not directly lead to estimated poses and uncertainties. Therefore, we develop RANdom SAmple averaGing (RANSAG) to compute an average pose and apply semidefinite relaxation to upper bound the worst-case errors between the average pose and the groundtruth. On the LineMOD Occlusion dataset we demonstrate: (i) the PURSE covers the groundtruth with valid probabilities; (ii) the worst-case error bounds provide correct uncertainty quantification; and (iii) the average pose achieves better or similar accuracy as representative methods based on sparse keypoints.

## Quick start

### Prepare data
- Download `data.zip` from this google drive [link](https://drive.google.com/file/d/1UGek7S3-4wwvgMlGvfBxJQPGW3Q2MfaR/view?usp=sharing)
- Unzip the data and put it into the `keypoint` folder (then you should have a folder `keypoint/data`)

### Conformal calibration

```python
python conformal_calibration.py --score_type ball
```

You can change `--score_type` to `ellipse` to use a different nonconformity function.
You can also add `--do_frcnn` to use FRCNN to detect object bounding boxes.

The calibration scores will be saved into a pickle file.

### Conformal prediction

```python
python conformal_prediction.py --score_type ball --epsilon 10 --save_fig
```
will write a set of pdf files drawing the conformal prediction sets (balls) into `keypoint/data/bop/lmo-org/icp_results`. You can change the results folder in `conformal_prediction.py`.

## Acknowledgement
The source code in the `keypoint` folder are adapted from the git repo https://github.com/yufu-wang/6D_Pose. We would like to thank Yufu Wang for helping us run the code.

## Citation
If you find this paper and implementation useful, please cite
```bibtex
@inproceedings{yang23cvpr-purse,
  title={Object pose estimation with statistical guarantees: Conformal keypoint detection and geometric uncertainty propagation},
  author={Yang, Heng and Pavone, Marco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8947--8958},
  year={2023}
}
```

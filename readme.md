# Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation

## Prepare data
- Download `data.zip` from this google drive [link](https://drive.google.com/file/d/1UGek7S3-4wwvgMlGvfBxJQPGW3Q2MfaR/view?usp=sharing)
- Unzip the data and put it into the `keypoint` folder (then you should have a folder `keypoint/data`)

## Conformal calibration

```
python conformal_calibration.py --score_type ball
```

You can change `--score_type` to `ellipse` to use a different nonconformity function.
You can also add `--do_frcnn` to use FRCNN to detect object bounding boxes.

The calibration scores will be saved into a pickle file.

## Conformal prediction

```
python conformal_prediction.py --score_type ball --epsilon 10 --save_fig
```
will write a set of pdf files drawing the conformal prediction sets (balls) into `keypoint/data/bop/lmo-org/icp_results`. You can change the results folder in `conformal_prediction.py`.
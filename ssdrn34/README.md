# SSD: Single-Shot MultiBox Detector (2016)

Original Paper: [arXiv](https://arxiv.org/pdf/1512.02325.pdf)

|Code Source|Link|License|
|:-|:-|:-|
| Original Caffe Source | [GitHub](https://github.com/weiliu89/caffe/tree/ssd) | BSD ([LICENSE.weiliu.ssd](../licenses/LICENSE.weiliu.ssd)) |
| MLPerf Reference Source (Inference) | [GitHub](https://github.com/mlcommons/inference/blob/r1.0/vision/classification_and_detection/python/models/ssd_r34.py) | Apache V2.0 ([LICENSE.mlperf.inference](../licenses/LICENSE.mlperf.inference)) |
| MLPerf Reference Source (Training) | [GitHub](https://github.com/mlcommons/training/tree/v0.5/single_stage_detector/ssd) | Apache V2.0 ([LICENSE.mlperf.training](../licenses/LICENSE.mlperf.training)) |
| Unofficial Impl (amdegroot) | [GitHub](https://github.com/amdegroot/ssd.pytorch) | MIT ([LICENSE.amdegroot.ssd](../licenses/LICENSE.amdegroot.ssd)) |
| Unofficial Impl (kuangliu) | [GitHub](https://github.com/kuangliu/pytorch-ssd) | MIT ([LICENSE.kuangliu.ssd](../licenses/LICENSE.kuangliu.ssd)) |
|||

## Getting Started
Download the COCO 2017 dataset using [MLPerf's download script](https://github.com/mlcommons/training/blob/v0.5/single_stage_detector/download_dataset.sh):
```bash
$ cd /path/to/coco
$ curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
$ curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
$ curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
```

## Examples
| Example | Description |
|-|-|
| `train_ssdrn34_coco.py` | Train the official MLPerf config for SSD-Resnet34 on COCO 2017. |
| `pred_ssdrn34_coco.py` | Predicts a few example images from the COCO 2017 Validation set. Dumps to tensorboard. |
| `list_coco_cats.py` | List categories in COCO 2017 |
| `stream_ssd.py` | PyQT App which reads webcam input and runs SSD-Resnet34 on the feed. |


How to run:
```bash
# Train SSDRN34
$ python -m ssdrn34.examples.train_ssdrn34_coco
```


## Validation
**Level 2**: In addition to being based on reference code (Level 1), the model in this repository has been studied to provide similar (eyeball validation) loss values when running on provided reference training data.

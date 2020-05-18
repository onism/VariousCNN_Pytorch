    ref:https://arxiv.org/pdf/2005.06831v1.pdf

# Abstract

We present a novel pipeline for semantic segmentation that detects out-of-distribution (OOD) segments by means of the DNN's prediction and performs image retrieval after feature extraction and dimensionality reduction on image patches.

# Introduction
Using a meta classification and regression approach termed MetaSeg to find unknown objects, we can group detected entities into visually and semantically related groups in order to enhance data exploration in the presence of domain shift.

Contribution:

1. We show that MetaSeg predicts the intersection over union of out of domain samples reliably.
2. Using MetaSeg we demonstrate that we are able to detect unknown object classes.
3. By extracting visual features we are able to group the found entities into an embedding space with semantically related neighborhoods
4. We perform an evaluation on the task of image retrieval with a variety of common deep learning architectures as feature extractors.

# OOD
Under the premise that objects of unknown classes mostly cause supicious predictions, we can auantify this effect and use it for OOD detection.

# Retrieval
After detecting image crops corresponding to segments with low estimated quality from newly collected data, further exploration of these crops can reveal weaknesses of the CNN with respect to the given domain shift.

In summary, our complete OOD detection and retrieval pipeline looks as follows:
1. Gather semantic segmentation predictions of newly collected samples
2. Rate all predicted segments according to their IoU estimated by MetaSeg
3. Detect segments with estimated IoU < 0.5 that are, however, still predicted to belong to classes that are high interest. For each candidate segment, the corresponding bounding box is used to provide a crop of the original input image.
4. Feed each crop through an embedding network pretrained on ImageNet and compute vectors of visual features.
5. (Optional) Reduce the dimensionality of the embedding space
6. Perform retrieval by NN search in the resulting embedding space.














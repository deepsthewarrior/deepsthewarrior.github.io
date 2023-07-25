#### Introduction:
Point Cloud data is how physical world is represented when we use sensors like LiDAR scanners, depth cameras, Photogrammetry(not a sensor), ToF Cameras. They contain x,y,z co-ordinate information of a point inside a Point Cloud. Developing deep learning models for Point Clouds is by itself a challenging task. We will discuss PointNet++ which is one of the pioneering works in this area.  PointNet++ was inspired from PointNet which introduced how to encode arbitary number of points to a fixed dimensional output representation. In order to understand PointNet++, we need to get hang of PointNet and the problems that it tried to solve.

**Properties of PointSet:**
Point set is a subset of data in a Point Cloud.
The 3 main properties of Point Sets are,

**Unordered**: 
Point set is different from image pixels or volumetric grid's voxels that it has no order.  Hence, the architecture we design must be able to learn the same from the permuted point sets also. 
	`PointNet uses maxpool as the symmetric funtion. Other examples of symmetric function are addition,multiplication(which are resilient to the input data ordering)`

**Interaction among Points**:
Learning the neighboring points near a given point, help in capturing the local contextual information, which helps the model to learn better. Similar characteristics is needed for image based deep learning models also.
`To solve this, PointNet concatenates the extracted global point features with each of the local point features, inside the segmentation Network. Thus being aware of both local and global information.`


<figure style="text-align:center;">
  <img src="/images/architecture.PNG" alt="PointNet architecture" />
  <p class="img-caption"> CNN at each level takes in the neighborhood points for calculating features at each level(Source: <a href="(https://www.baeldung.com/cs/cnn-receptive-field-size)">[http://stanford.edu/~rqi/pointnet2](https://www.baeldung.com/cs/cnn-receptive-field-size)/</a>)</p>
</figure>
**Invariance Under Transformations**
The model should be able to learn even if the point cloud undergoes rotation, transformation etc., This should not affect the classification category or the segmentation.
`A joint Alignment Network is introduced for this purpose.`

#### PointNet++
As mentioned above, PointNet does not efficiently capture the local features like how CNN  does in an hierarchical manner, with neighborhood information. In a CNN a set of neighbouring features contribute to each feature at the higher level.  Well, in PointNet the local features are only mapped and concatenated to the corresponding global features, not efficiently learning the neighborhood information.

PointNet++ groups the points based on distance metric and extracts the features from grouped points. 
<figure style="text-align:center;">
  <img src="/images/receptive.PNG" alt="Receptive Field of CNN" />
  <p class="img-caption"> CNN at each level takes in the neighborhood points for calculating features at each level(Source: <a href="(https://www.baeldung.com/cs/cnn-receptive-field-size)">(https://www.baeldung.com/cs/cnn-receptive-field-size)/</a>)</p>
</figure>
##### Hierarchical PointSet Feature Learning:
The hiearchical learning regime in PoinNet is comprised of multiple set abstraction operation. 

###### Set Abstraction:
Input to the set abstraction Operation is N x (d+C) matrix, where N is the number of points, d is the number of dimension in each point and C dimension of point features. It outputs a matrix of dimension N' x(d+C') where N' is the number of subsampled points and C' is the dimension of features after it goes through feature extraction.

Now, let us see what each Set abstraction layer is composed of:
1. **Sampling Layer:**
   Sampling operation selects m points from n given input points. Here we use Farthest point Sampling as the sampling algorithm. First it selects a point randomly and selects the second point which farthest away from the first point in terms of distance metric. The selected points are a part of selected point set. Then it selects the 3rd point which far away from points in the selected point set. The selected point now constitutes a part of selected point set. Thus, it proceeds until the selected point set has m points. 
   
<figure style="text-align:center;">
  <img src="/images/fps.PNG" alt="Farthest Point Sampling of a Chair object" />
  <p class="img-caption">Blue dots represent the raw points, Violet dots represents the sampled points(Source: <a href="https://arxiv.org/pdf/1812.01659.pdf">https://arxiv.org/pdf/1812.01659.pdf</a>)</p>
</figure>


2. **Grouping Layer:**
   Each sampled point is considered as the centroids  and the ball query operation is used to gather all points within the radius of K around the centroid. Similar to CNN, centroids are mapped to the points neighboring it. 
  *Input:* N x (d+C) point set, N'xd centroids
  *Output:* Point Set groups N' x K x (d+C) where K is the number of neighborhood points to a centroid.
  
   
Source:https://link.springer.com/chapter/10.1007/978-3-030-61864-3_27

<figure style="text-align:center;">
  <img src="/images/ball_query.PNG" alt="Ball Query operation" />
  <p class="img-caption">(Source: <a href="(https://link.springer.com/chapter/10.1007/978-3-030-61864-3_27)">(https://link.springer.com/chapter/10.1007/978-3-030-61864-3_27)</a>)</p>
</figure>

Function implementing Grouping and Sampling as per the Offical Code Repo:
``` python
def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):

'''
Input:

npoint: int32

radius: float32

nsample: int32

xyz: (batch_size, ndataset, 3) TF tensor

points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points

knn: bool, if True use kNN instead of radius search

use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features

Output:

new_xyz: (batch_size, npoint, 3) TF tensor

new_points: (batch_size, npoint, nsample, 3+channel) TF tensor

idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points

grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs

(subtracted by seed point XYZ) in local regions

'''

  

new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)

if knn:

_,idx = knn_point(nsample, xyz, new_xyz)

else:

idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)

grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization

if points is not None:

grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)

if use_xyz:

new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)

else:

new_points = grouped_points

else:

new_points = grouped_xyz

  

return new_xyz, new_points, idx, grouped_xyz 
```
3. **PointNet Layer:**
   The input to the pointnet is  *N'* local regions with K neighbouring points of dimension (d+C) for the centroid inside each local region. So it accounts to *N' x K x (d+C)* dimension.
   
   The co-ordinates of the neighboring points are changed to centroid as its origin from the global co-ordinate system. In other words each point in the local region will have its co-ordinates depicting how far it is from the centeroid rather than how far it is from the global origin.

   The property of PointNet is that given any number of points as input(K points in our case), it gives *1 x C'* dimensional vector as output. In our context, note that K can be a varying number for each local region. Thus for *N'* local regions, *N' x (d+C')* as the output. 

<figure style="text-align:center;">
  <img src="/images/pointnet.PNG" alt="PointNet++ architecture" />
  <p class="img-caption">(Source: <a href="http://stanford.edu/~rqi/pointnet2/">http://stanford.edu/~rqi/pointnet2/</a>)</p>
</figure>

The overall set abstraction function with MSG code block is here
``` python
def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):

''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)

Input:

xyz: (batch_size, ndataset, 3) TF tensor

points: (batch_size, ndataset, channel) TF tensor

npoint: int32 -- #points sampled in farthest point sampling

radius: list of float32 -- search radius in local region

nsample: list of int32 -- how many points in each local region

mlp: list of list of int32 -- output size for MLP on each point

use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features

use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format

Return:

new_xyz: (batch_size, npoint, 3) TF tensor

new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor

'''

data_format = 'NCHW' if use_nchw else 'NHWC'

with tf.variable_scope(scope) as sc:

new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))

new_points_list = []

for i in range(len(radius_list)):

radius = radius_list[i]

nsample = nsample_list[i]

idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

grouped_xyz = group_point(xyz, idx)

grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])

if points is not None:

grouped_points = group_point(points, idx)

if use_xyz:

grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)

else:

grouped_points = grouped_xyz

if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])

for j,num_out_channel in enumerate(mlp_list[i]):

grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],

padding='VALID', stride=[1,1], bn=bn, is_training=is_training,

scope='conv%d_%d'%(i,j), bn_decay=bn_decay)

if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])

new_points = tf.reduce_max(grouped_points, axis=[2])

new_points_list.append(new_points)

new_points_concat = tf.concat(new_points_list, axis=-1)

return new_xyz, new_points_concat
```
##### Non -Uniform Sampling Density:
Point cloud does not have uniform distribution of points in all regions. Some regions have dense points and some have sparse points. This is a very common characteristic when the point data is captured from real world. 
Models trained on sparse point clouds may not capture finer local details and if the model is learning on dense regions more, then it cannot generalize for sparse data. As a hack around this, we need to learn on larger areas in sparse regions and need to learn on smaller areas for dense regions. In simple words, we need to learn in a density adaptive manner. 

We adjust the Set Abstraction operation accordingly to facilitate the density adaptive feature learning.

##### Multi-Scale Grouping:
During grouping operation, we apply ball query of different Radius around a centroid. We then seperately learn features on different scales. In the end we concatenate them to obtain Multi Scale features. 

<figure style="text-align:center;">
  <img src="/images/msg.PNG" alt="Multi Scale Groupping" />
  <p class="img-caption">(Source: <a href="http://stanford.edu/~rqi/pointnet2/">http://stanford.edu/~rqi/pointnet2/</a>)</p>
</figure>
But how do they compensate for the extra points that we sampled? They use random input dropping strategy, where each input point is dropped with a probability $\theta$.  This method is used only during training. During testing, all points are utitlized.

##### Multi-resolution grouping (MRG):
In Multi Resolution grouping we concatenate 2 vectors. The first one learns from the features at one level lower to it(vector on left side). While the second one learns from the features directly at the point level(vector on the right side.)

When the point cloud is sparse, first vector is weighed low. This owing to the fact that the first vector has poor sampling as it is being recursively sampled on sparse regions. In this case the second vector has sampled more points.

When the point cloud is dense the first vector is weighed high, as the features learnt have fine contextual information due to recursive learning like the receptive field of CNN. 

<figure style="text-align:center;">
  <img src="/images/msr.PNG" alt="Multi Resolution Groupping" />
  <p class="img-caption">(Source: <a href="http://stanford.edu/~rqi/pointnet2/">http://stanford.edu/~rqi/pointnet2/</a>)</p>
</figure>

This takes us to the end of feature extractor part.

##### Point Feature Propagation for Set Segmentation:

For segmentation tasks, we need to get point features for all raw points. Feature propagation is used to propagate points using interpolation and skip connections. As given in the Fig representing the architecture, the output points after a Set Abstraction(SA) operation are interpolated to N the dimension of N points which were input to the SA operation. Then the original N features are also concatenated in a skip link concatenation manner. This loosely similar to the U-Net architecture, where we reconstruct and also concatenate the features from the encoder to the corresponding levels in the decoder. 
Then the concatenated features are fed to Unit PointNet containing Conv layers. 
Interpolation, concatenation and Unit PointNet combine together to form a block. Each block is repeated multiple times until we reach the size of the raw point clouds, on which label is generated pointwise by the unit pointnet.
<figure style="text-align:center;">
  <img src="/images/segmentation.PNG" alt="Segmentation Task" />
  <p class="img-caption"> Interpolate, Skip link concatenation, Unit PointNet form (Source: <a href="http://stanford.edu/~rqi/pointnet2/">http://stanford.edu/~rqi/pointnet2/</a>)</p>
</figure>

For Classification task, we send the extracted features into the Fully Connected Layers to obtain class scores.

<figure style="text-align:center;">
  <img src="/images/classification.PNG" alt="Segmentation Task" />
  <p class="img-caption"> Classification Network (Source: <a href="http://stanford.edu/~rqi/pointnet2/">http://stanford.edu/~rqi/pointnet2/</a>)</p>
</figure>

#### Conclusion:
PointNet++, since its release has paved way for other powerful models models like PointRCNN, PVRCNN. It is till today one of the most relevant feature extractors on PointClouds


#### References:
1. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
2. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
3. https://github.com/charlesq34/pointnet2/

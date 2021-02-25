# d2l目标检测笔记

## 边界框(Bounding Box)

### 表示方法

​	在目标检测里，我们通常使用边界框（bounding box, 缩写为bbox）来描述目标位置。

​	边界框是一个矩形框，可以由矩形**左上角**的x和y轴坐标与**右下角**的x和y轴坐标确定$(x_1, y_1, x_2, y_2)$。另外，也可以用边界框中心的x和y轴坐标及其宽度w和高度h表示$(x, y, w, h)$。两种表示方法可以进行相互转化：

```python
# codes from d2l
# The input argument boxes can be either a length  4  tensor, or a  (N,4)  2-dimensional tensor.
def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper_left, bottom_right)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

### 可视化函数

为了能在图中显示出边界框，可以使用`plt`的`Rectangle`函数。若坐标为左上右下表示法，则可以用下面的函数来画边界框：

```python
# codes from d2l
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

该函数还未能显示标签信息，关于显示标签信息的函数`show_bboxes`，将在锚框一节中给予说明。

## 锚框(Anchor Boxes)

### 简介

目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边缘从而更准确地预测目标的真实边界框（ground-truth bounding box）。不同的模型使用的区域采样方法可能不同，其中的一种方法是：以**每个像素为中心**生成多个大小（size）和宽高比（aspect ratio）不同的边界框。这些边界框被称为锚框（anchor box）。

### 锚框的生成

锚框的坐标依然可以用bbox中的表示方法。一般来说，锚框的坐标会用**归一化**的形式来表示，值域为[0,1]，后续乘以图像的高宽来得到真正的坐标。使用归一化表示后，这样的坐标可以用**两个指标**来计算，即上面提到的size和aspect ratio。其中size衡量的是锚框相对于原图的大小(其中$s^2$为归一化锚框面积)，值域为(0,1]；ratio衡量的是归一化坐标中的宽高比，r=1表示与原图宽高比例一致（并非表示正方形）。

有了这两个指标，只要给定一个像素坐标并以此为中心（中心宽高表示法），归一化的宽与高可以分别计算如下：
$$
w' = s \sqrt{r} \\
h' = \frac{s}{\sqrt{r}}
$$
真实的宽与高为：
$$
w_{anchor} = w_{img}\times w' \\
h_{anchor} = h_{img}\times h'
$$
由此可以得出锚框的四个坐标(x,y,w,h)。

一般来说，在生成锚框时，会设置多个size与ratio。假设有m个size与n个ratio，若产生所有的锚框，则有$hwmn$个（hw为像素总个数），这会大大增加计算复杂度。因此，通常的做法是只取size的第一个与所有ratio组合，然后再取ratio的第一个与所有size组合，从而在一个像素中心产生m+n-1个锚框，大大降低锚框数量，起到提高效率的作用。

目前anchor box的size与ratio选择主要有三种方式：

- 人为经验选取
- k-means聚类
- 作为超参数进行学习

在d2l一书中，作者将锚框坐标转化为归一化的左上右下表示法，函数`multibox_prior`如下：

```python
def multibox_prior(data, sizes, ratios):
    '''
    data: tenseor [batch_size, channels, h, w] of fmaps
    output: (batch size, number of anchor boxes, 4) 
    use output.reshape(batch_size, h, w, #anchor boxes of each pixel, 4) to get the coordinates.
    '''
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to center of a pixel
    # Since pixel (height=1, width=1), we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate boxes_per_pixel number of heights and widths which are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    # cat (various sizes, first ratio) and (first size, various ratios)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have boxes_per_pixel number of anchor boxes, so
    # generate grid of all anchor box centers with boxes_per_pixel repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

# Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
```

为了展示一个像素中心产生的所有锚框，作者又写了一个锚框的可视化函数`show_bboxes`，利用`plt`的`axes.text`函数添加了标签来展示size与ratio。这样的操作可以借鉴于展示类别标签与置信度。

```python
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.
    axes: plt.axes
    bboxes: [#anchor boxes, 4]
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0],
                      rect.xy[1],
                      labels[i],
                      va='center',
                      ha='center',
                      fontsize=9,
                      color=text_color,
                      bbox=dict(facecolor=color, lw=0))
 
# bbox_scale = torch.tensor((w, h, w, h))
# fig = plt.imshow(img)
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
```

![../_images/output_anchor_f592d1_53_0.svg](https://d2l.ai/_images/output_anchor_f592d1_53_0.svg)

### 交并比(IoU)

为了衡量锚框识别物体的好坏，引入一个指标——交并比(IoU, intersection over union), 两个边界框相交面积与相并面积之比。书中的这一图片直观的展示了它的计算方法：

![交并比是两个边界框相交面积与相并面积之比](https://zh.d2l.ai/_images/iou.svg)



在编程中，可以先分别计算两个区域的面积，再计算相交面积，最后的相并面积为两个区域面积之和减去相交面积。作者用下面的函数`box_iou`来计算IoU：

```python
def box_iou(boxes1, boxes2):
    """Compute IOU between two sets of boxes of shape (N,4) and (M,4).
    	Return a N*M tensor of IoU value
    """
    # Compute box areas
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    unioun = area1[:, None] + area2 - inter
    return inter / unioun
```

### 锚框的标注

在训练集中，我们将每个锚框(anchor box)视为一个训练样本。为了训练目标检测模型，我们需要为每个锚框标注两类标签：一是锚框所含目标的类别，简称**类别**；二是真实边界框（ground-truth bounding box, gt bbox）相对锚框的偏移量，简称**偏移量（offset）**。在真正应用目标检测模型时，我们首先生成多个锚框，然后为每个锚框预测类别以及偏移量，接着根据预测的偏移量调整锚框位置从而得到预测边界框，最后筛选需要输出的预测边界框。这便是目标检测的基本步骤。由此看出，我们还需要对锚框进行标注，标注方法如下：

1. 计算锚框与gt bbox的IoU

   假设总的锚框数为$n_a$，gt bbox的数量为$n_b$($n_a \geq n_b$)，那么通过他们的计算IoU，可以得到一个$n_a \times n_b$的矩阵$I$。

2. 优先标注$n_b$个锚框

   首先取矩阵中最大的元素，设为$I_{ij}$，将第i个锚框标注分配到第j个gt bbox，**得到类别并计算偏移量**。随后丢弃矩阵$I$的第i行与第j列，再找剩余元素中的最大值，重复该过程直到所有gt bbox都有锚框匹配。

   ![为锚框分配真实边界框](https://zh.d2l.ai/_images/anchor-label.svg)

3. 标注剩余锚框

   取其IoU最大的一个gt bbox，若IoU**大于某一阈值**（如0.5），那么就将两者匹配，得到类别并计算偏移量。如果一个锚框没有被分配gt bbox，则将该锚框的类别设为背景。类别为背景的锚框通常被称为负类锚框，其余则被称为正类锚框。

类别的标注很好理解，只需与bt bbox一致。偏移量的计算比较特殊，由于数据集中各个框的位置和大小各异，因此这些相对位置和相对大小通常需要一些特殊变换，才能使偏移量的分布更均匀从而更容易拟合。其中一个公式是：
$$
\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),
$$
其中b表示gt bbox，a表示锚框，常数值 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1, and\ \sigma_w=\sigma_h=0.2$.（经验值）

下面的函数`multibox_target`包含了锚框标注的过程，其中`offset_boxes`计算了偏移量：

```python
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

#@save
def multibox_target(anchors, labels):
    '''
    anchors: (1, #anchor boxes, 4)
    labels: (batch_size, #gt bbox, 4)
    Return:
    bbox offset: (batchsize, 4*#class) background: 0  target: offset(x, y, w, h)
    bbox mask: (batchsize, 4*#class) 0:background(neg) 1:target(pos)
    class label: (batchsize, #class) 0:background  1: class 1  2: ...
     '''
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = match_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # Initialize class_labels and assigned bbox coordinates with zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Assign class labels to the anchor boxes using matched gt bbox labels
        # If no gt bbox is assigned to an anchor box, then let the
        # class_labels and assigned_bb remain zero, i.e the background class
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # offset transformations
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

返回的`bbox_mask`主要是为了过滤负类样本，使其不影响目标函数的计算。

### 预测框的输出

在模型预测阶段，我们先为图像生成多个锚框，并为这些锚框一一预测类别和偏移量。随后，我们根据锚框及其预测偏移量得到预测边界框。由预测偏移量生成预测bbox坐标的函数`offset_inverse`，相当于计算offset的逆运算：

```python
# code from d2l
def offset_inverse(anchors, offset_preds):
    c_anc = d2l.box_corner_to_center(anchors)
    c_pred_bb_xy = (offset_preds[:, :2] * c_anc[:, 2:] / 10) + c_anc[:, :2]
    c_pred_bb_wh = torch.exp(offset_preds[:, 2:] / 5) * c_anc[:, 2:]
    c_pred_bb = torch.cat((c_pred_bb_xy, c_pred_bb_wh), axis=1)
    predicted_bb = d2l.box_center_to_corner(c_pred_bb)
    return predicted_bb
```

当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。为了使结果更加简洁，我们可以移除相似的预测边界框。常用的方法叫作非极大值抑制（non-maximum suppression，NMS）。

NMS的主要思想如下：

1. 算法准备：对于一个预测边界框B，模型最终会输出会计算它属于每个类别的概率值，其中概率值最大对应的类别就是预测边界框的类别。

2. 在同一图像上，把所有预测边界框(不区分类别)的预测概率从大到小进行排列，然后取出最大概率的预测边界框$B_1$作为基准，然后计算剩余的预测边界框与$B_1$的交并比，如果大于给定的某个阈值(如0.5)，则将这个预测边界框移除。

3. 从剩余的预测边界框中选出概率值第二最大的预测边界框$B_2$，计算过程重复上述的过程，直到所有的边界框都曾选作基准。

通过NMS算法，可以滤除许多相似的锚框，从而得到准确率较高且没有太多重叠的预测框。

下面的函数`nms`实现了NMS算法：

```python
def nms(boxes, scores, iou_threshold):
    '''
    boxes:(#boxes, 4)
    scores:(#boxes, )
    Return:
    (#boxes that are left, 4)
    '''
    # sorting scores by the descending order and return their indices
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # boxes indices that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)
```

再利用`multibox_detection`函数封装该`nms`函数，从而实现所有锚框的标记。（被nms删除的boxes标记为背景，标签为-1）

```python
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.00999999978):
 	'''
 	cls_probs: (batch_size, #classes+1(bg), #boxes)
 	offset_preds: (batch_size, #anchors*4)
 	anchors: (1, #anchors, 4)
 	pos_threshold: pos锚框的阈值
 	Return:
 	(batch size, #anchors, 6)
 	6: 0-class label(0,1,..., -1(bg)) 1-conf, 2-5:(x1,y1, x2,y2) predicted bbox(range: 0-1)
 	'''
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, 0.5)
        # Find all non_keep indices and set the class_id to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # threshold to be a positive prediction
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
```

利用上述函数，过滤掉标签为背景的预测框，即可得到最终预测的边界框。

另外，作者告诉我们一个技巧。实践中，我们可以在执行非极大值抑制前将置信度较低的预测边界框移除，从而减小非极大值抑制的计算量。我们还可以筛选非极大值抑制的输出，例如，只保留其中置信度较高的结果作为最终输出。

## 多尺度检测(Multiscale Object Detection)

如果以输入图像每个像素为中心都生成锚框，很容易生成过多锚框而造成计算量过大。减少锚框个数并不难。一种简单的方法是在输入图像中均匀**采样一小部分像素**，并以采样的像素为中心生成锚框。采样的方法通常是在卷积网络后生成的feature map(fmap)中生成锚框。由于锚框坐标为归一化坐标，尽管在fmap中每个像素都采样，但映射到原图像后，采样是均匀间隔分布的。而且可以在多个fmap中进行采样，从而得到不同尺度的锚框，实现多尺度检测。

```python
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
  # display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

fmap为4*4，生成16处的锚框，可以检测小物体：

![../_images/chapter_computer-vision_multiscale-object-detection_5_0.svg](https://zh.d2l.ai/_images/chapter_computer-vision_multiscale-object-detection_5_0.svg)

fmap为1*1，生成1处的锚框，可以检测大物体：

![../_images/output_multiscale-object-detection_ad7147_42_0.svg](https://d2l.ai/_images/output_multiscale-object-detection_ad7147_42_0.svg)

另外，在某个尺度下，假设$c_i$张特征图(channels?)为卷积神经网络根据输入图像做前向计算所得的中间输出。特征图在相同空间位置的$c_i$个单元在输入图像上的感受野相同，并表征了同一感受野内的输入图像信息。 因此，我们可以将特征图在相同空间位置的$c_i$个单元变换为以该位置为中心生成的a个锚框的类别和偏移量。

## 参考资料

- <https://d2l.ai/chapter_computer-vision/bounding-box.html>
- <https://d2l.ai/chapter_computer-vision/anchor.html#generating-multiple-anchor-boxes>
- <https://zhuanlan.zhihu.com/p/63024247>




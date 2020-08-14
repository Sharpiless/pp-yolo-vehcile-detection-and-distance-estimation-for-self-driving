# 基于PP-yolo完成自动驾驶中车辆检测、车距估计和红绿灯识别：

PaddleDetection：https://github.com/PaddlePaddle/PaddleDetection

我的博客：https://blog.csdn.net/weixin_44936889

论文地址：[https://arxiv.org/pdf/2007.12099.pdf](https://arxiv.org/pdf/2007.12099.pdf)

论文详解：https://blog.csdn.net/weixin_44936889/article/details/107560168

# 最终效果：

其中检测框上面的数值是一个相对的距离值

![a](https://img-blog.csdnimg.cn/2020081422393128.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70#pic_center)



# 简介：
目标检测是计算机视觉研究的重要领域之一，在各种实际场景中起着至关重要的作用。在实际应用中，由于硬件的限制，往往需要牺牲准确性来保证检测器的推断速度。因此，必须考虑目标检测器的有效性和效率之间的平衡。本文的目标不是提出一种新的检测模型，而是实现一种效果和效率相对均衡的对象检测器，可以直接应用于实际应用场景中。考虑到YOLOv3在实际应用中的广泛应用，作者开发了一种新的基于YOLOv3的对象检测器。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200724142318338.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

本文的作者们主要尝试**结合现有的几乎不增加模型参数和FLOPs的各种技巧**，在保证速度几乎不变的情况下，尽可能提高检测器的精度。由于本文所有实验都是基于PaddlePaddle进行的，所以作者称之为PP-YOLO。通过多种技巧的结合，PP-YOLO可以在效率(45.2% mAP)和效率(72.9 FPS)之间取得更好的平衡，超过了目前最先进的检测器EfficientDet和YOLOv4。

并且不像YOLOv4花费很大努力在探索高效的主干网络和数据增强策略，并且使用了NAS来搜索超参数（使得模型泛化性降低），PP-YOLO探索了更高效、使得模型更具有泛化性的策略。它的主干网络仅仅使用了ResNet，数据增强也只是直接用了MixUp，这使得PP-YOLO在训练和推理是更加高效。

同时这也说明使用更多Tricks和最新的主干网络，可能会使得PP-YOLO模型更准确。

# 网络结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200724143307801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

### 简介

由于项目需要目标追踪，之前打算是用OpenCV提供的算法进行追踪，但实测下来效果不是很理想，了解到使用YOLOv5与DeepSort相结合的方式可以进行多物体追踪，在跑通作者提供的track示例后，将目标检测和目标追踪分别进行封装，方便以后在项目中使用。

### 原始track介绍

我们阅读track.py中代码，从中可以看出来代码包含以下部分

- 参数配置（YOLOv5和DeepSort）
- 初始化DeepSort（DeepSort）
- 加载检测model（YOLOv5）
- 进行推理，获得相关信息（YOLOv5）
- 进行追踪（DeepSort）
- 结果显示的处理（YOLOv5和DeepSort）

按照上述流程我们可以对检测和追踪分别进行封装。

#### 参数配置

主方法中关于`parser`的所有语句都是加载命令行参数【219-241行】，我们如果要应用到项目中很少会用到命令行提供参数，所以我们需要写一个能提供参数的方法或类。

`detect`方法中刚开始就是对YOLOv5和DeepSort的相关参数进行配置【47-55行】

#### 初始化DeepSort

【54-62行】是获得DeepSort的配置文件，并构造deepsort对象

#### 加载检测model

【77-104行】加载相关的model

#### 进行推理

对以及打包的原始图像进行处理【91-95行】（对图片信息打包）【113-118行】（对图片进行浮点数、归一化处理），然后进行推理，之后进行非极大抑制，对所得的结果进行处理【129-150行】

#### 进行追踪

调用`update`方法进行追踪。【154】

注意！ 追踪更新的返回 结果都是tensor

#### 最终展示结果处理

对获得的物体的ID、类别名称（cls）、置信度（confidence）、boundingboxes（xyxy）进行最终处理【157-167】

#### ！

对于保存视频、文件等相关代码本文未讲述到。

### 如何运行

1. 克隆此项目

    ```shell
    git clone https://github.com/jimyag/YOLOv5-DeepSort.git
    ```

2. 安装依赖

    在终端执行以下语句安装依赖包

    `pip install -r requirements.txt`

3. 下载权重文件

    1. 下载DeepSort权重

        [点击下载](https://github-releases.githubusercontent.com/275118967/8c1c5d80-cf4e-11eb-8c2e-40921f433dff?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210804%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210804T162651Z&X-Amz-Expires=300&X-Amz-Signature=6ee3ee2f893851dcf9eea7f645327ca61351e8ba432187d22ceff5428802bf3c&X-Amz-SignedHeaders=host&actor_id=69233189&key_id=0&repo_id=275118967&response-content-disposition=attachment%3B%20filename%3Dckpt.t7&response-content-type=application%2Foctet-stream)，并将下载好的权重文件放在`deep_sort_pytorch/deep_sort/deep/checkpoint`下。

        如果要使用其他权重，需要修改`deep_sort_pytorch/configs/deep_sort.yaml`中`REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"`的配置文件

    2. 下载YOLOv5权重

        [点击下载](https://github-releases.githubusercontent.com/264818686/56dd3480-9af3-11eb-9c92-3ecd167961dc?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210804%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210804T163304Z&X-Amz-Expires=300&X-Amz-Signature=970d9bbd047c39cff68107f4367a4f8a5da6a82de8cce3679042c3b7ae97286a&X-Amz-SignedHeaders=host&actor_id=69233189&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dyolov5s.pt&response-content-type=application%2Foctet-stream)，并将下载好的权重文件放在`yolov5/weights`,并修改`my_detect.py`中`self.yolo_weights = 'yolov5/weights/basketball_robot.pt'`

4. 更改源文件

    在`my_detect.py`中修改`self.source = 'basketball.mp4'`






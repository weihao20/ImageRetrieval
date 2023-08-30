# 针对全局和人脸特征的图像检索方法

## 1.算法描述

针对电影图像检索场景，提出针对全局和人脸特征的图像检索方法，分别根据查询图像的全局特征和图像中的人脸特征，在电影图像数据库中进行检索，从而提升了图像检索的准确度。

## 2. 环境依赖及安装

所有实验都在一台配备了 ubuntu 16.04操作系统、Intel(R) Xeon(R) Gold 5218 处理器和 NVIDIA GeForce RTX 3090 显卡的服务器上进行，相关依赖如下：

- Python 3.8.0

- Pytorch 1.10.2

- NumPy 1.22.3

建议执行如下命令，安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 运行示例

### 数据集

方法应用于项目中的电影细粒度数据集，该数据集存储于项目服务器中。

### 数据预处理

首先对数据库中的图像进行预处理，提取全局特征：

```bash
python retrieval_image/make_database.py
```

生成的图像特征和路径信息储存在`retrieval_image/database`。

### 测试

运行 `demo.py` 来测试基于全局特征和人脸特征的图像检索。

对于全局特征，图像检索方式为：

```python
from retrieval_image import ClipRetrieve
model = ClipRetrieve()
results = model.i2i_match(query_path='frame.jpg')
```

对于人脸特征，图像检索方式为：

```python
from facev1 import faceRetrieve
model = faceRetrieve(base_path='facev1/database', 
                     yunet_model='facev1/models/face_detection_yunet_2022mar.onnx',
                     sface_model='facev1/models/face_recognition_sface_2021dec.onnx')
results = model.retrieve(query_path='frame.jpg', topk=2)
```

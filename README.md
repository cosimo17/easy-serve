
![logo](assets/logo.jpg)
## 介绍
一个简单易用的深度学习模型部署框架    
特性:
+ 支持多GPU多进程
+ 支持batch推理
+ 基于aiohttp，支持高性能的异步并发
+ 配置灵活，与推理引擎解耦
+ 开箱即用的Docker支持
+ 进程守护，崩溃自动重启

## 安装与配置
首先，克隆本项目的源代码至本地    
`git clone https://github.com/cosimo17/easy-serve.git`    
安装依赖包
`pip install -r requirements.txt`    
## 运行示例
提供了Resnet18图像分类和yolov3目标检测两个示例供参考和测试。
请先下载对应的权重至models目录下。    
[Resnet18权重](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx)    
[Yolov3权重](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx)    
启动服务器    
`python src/app.py -c configs/imagenet_resnet18.yaml`    
运行客户端进行测试    
`python src/sample_client.py`    

## 单机部署
自定义handle    
根据自己的模型和业务逻辑编写对应的handle, handle负责处理深度学习模型的前向推理(forward)，输入预处理(preprocess)以及输出结果的后处理(postprocess)，所有和模型相关的操作等应该封装在handle中,
参考handle.py中的示例代码    

修改配置文件    
使用yaml文件来进行配置，参考configs/imagenet_resnet18.yaml.    
参数说明如下:
+ GPUs, 字符串类型, 指定要使用的GPU的编号。要使用多个GPU时，中间以逗号隔开, "0,1,2"
+ ProcessPerGPU: 整数型，表示需要在一个GPU上加载多少个模型实例，当模型本身较小时，为了充分利用显卡的算力和显存，一张GPU上可以同时加载多个模型实例，如果设置过大，可能导致显存不足（OOM）
+ Handle: 指定handle，handle名应与handle.py中自定义的handle的类名相同    

定义好handle，编写完配置文件后，就可以启动部署    
`python src/app.py -c configs/your_config.yaml`    

数据格式    
客户端和服务器端使用json对象来传递数据，json对象会被解析为一个字典，服务器端拿到客户端传递的json字典后，会对其添加一些必要的字段。
假设客户端传递的字典为    
`{'img': abcde}`    
经过服务器端包装后，传递给handle的字典格式变为    
`{'task_id': xxxxx, 'task_info': {'img': abcde}}`
## 多机分布式部署
![分布式部署](assets/分布式部署概念图.png)    
使用nginx与本项目结合，可以方便地进行多机分布式部署    
安装nginx    
`sudo apt-get install nginx`

配置nginx    
创建一个文本文件，将下面的内容复制粘贴进去，修改upstream中的服务器配置，修改listen的端口号。将文本文件移动到/etc/nginx/sites-enabled    
修改/etc/nginx/nginx.conf，在http一节的大括号的末尾添加“include /etc/nginx/sites-enabled/你的文本文件名;”
```
upstream inference {
    server ip_of_serve1:port;
    server ip_of_server2:port;
    # other serves
}

server {
    listen 8080;
    server_name ml.easy-serve.com;

    location / {
        proxy_pass http://inference;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
启动nginx    
`sudo systemctl -l enable nginx && sudo service nginx start`
## 进程守护
watchdog.py实现了一个简单的监视器，可以监控服务器进程的运行状态并在服务器进程崩溃或推出时自动重启。    
如果使用watchdog的方式启动，就不需要在手动启动app.py，watchdog运行时会自动启动服务器进程。    
`python watchdog.py -c configs/your_config.yaml`

## 常见问题
### 什么是深度学习模型部署框架
深度学习模型训练完成后，一般有两种部署方式:端侧部署和服务器侧部署。服务器侧部署即将模型部署为一个web应用，允许通过http请求进行访问。
将服务器侧部署所需的一些共同功能抽象出来，允许用户通过配置来自定义地部署自己的模型，这就是框架的功能。
### 一个GPU上加载多少个模型实例最好
一个GPU上加载多少个模型实例最好，由GPU的显存大小和算力决定。加载的模型实例越多，越能完全利用GPU的显存和算力，相应地也会需要更多的资源。
在GPU的显存或者算力饱和之前，加载更多的实例可以提高模型的吞吐率，加载的模型实例太多，可能会导致GPU显存溢出(out of memory)。可以先设置一个较小的值，然后进行测试，观察GPU的显存占用量和计算利用率，根据实际情况逐步增加实例数量。

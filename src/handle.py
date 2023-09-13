from abc import ABCMeta, abstractmethod
import onnxruntime as ort
import cv2
import numpy as np
import base64


class ClassRegistry:
    def __init__(self):
        self.class_dict = {}

    def reg(self):
        def decorator(cls):
            self.class_dict[cls.__name__] = cls
            return cls

        return decorator

    def create_instance(self, name):
        if name in self.class_dict:
            return self.class_dict[name]()
        else:
            raise KeyError(f"Class with name '{name}' not found in the registry")


registry = ClassRegistry()


class BaseHandle(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, data):
        raise NotImplemented

    @abstractmethod
    def forward(self, data):
        raise NotImplemented

    @abstractmethod
    def postprocess(self, data):
        raise NotImplemented

    @abstractmethod
    def __call__(self, data):
        raise NotImplemented


# Example. Use onnxruntime to execute the resnet18
@registry.reg()
class ImagenetResnet18Handle(BaseHandle):
    def __init__(self, gpu):
        model_path = '../models/resnet18-v1-7.onnx'
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': int(gpu),
            }),
            'CPUExecutionProvider',
        ]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.output_names = ['resnetv15_dense0_fwd']
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape((1, 1, 3))

    def preprocess(self, data):
        image_base64 = data['task_info']['img']
        decoded_image_data = base64.b64decode(image_base64)
        decoded_image = cv2.imdecode(np.frombuffer(decoded_image_data, np.uint8), cv2.IMREAD_COLOR)
        decoded_image = cv2.resize(decoded_image, (224, 224))

        decoded_image = decoded_image.astype(np.float32)
        img = (decoded_image - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        data['task_info']['img'] = img
        return data

    def forward(self, data):
        output = self.session.run(output_names=self.output_names, input_feed={'data': data['task_info']['img']})
        data['task_info']['result'] = output
        del data['task_info']['img']
        return data

    def postprocess(self, data):
        result = data['task_info']['result']
        result = result[0]
        result = np.argmax(result, axis=1).tolist()
        data['task_info']['result'] = result
        return data

    def __call__(self, data):
        return self.postprocess(self.forward(self.preprocess(data)))


# Example. Use onnxruntime to execute the yolov3-coco
@registry.reg()
class CoCoYOLOV3Handle(BaseHandle):
    def __init__(self, gpu):
        model_path = '../models/yolov3-10.onnx'
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': int(gpu),
            }),
            'CPUExecutionProvider',
        ]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = "input_1"
        self.output_names = ['yolonms_layer_1/ExpandDims_1:0', "yolonms_layer_1/ExpandDims_3:0",
                             "yolonms_layer_1/concat_2:0"]
        self.input_size = (416, 416)
        self.score_threshold = 0.5
        self.object_classes = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    @staticmethod
    def letterbox_image(image, size):
        h, w, _ = image.shape
        target_h, target_w = size

        # Calculate the scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        letterboxed_image = np.full((target_h, target_w, 3), (128, 128, 128), dtype=np.uint8)

        # Calculate the position to paste the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Paste the resized image onto the blank canvas
        letterboxed_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        return letterboxed_image

    def preprocess(self, data):
        image_base64 = data['task_info']['img']
        decoded_image_data = base64.b64decode(image_base64)
        decoded_image = cv2.imdecode(np.frombuffer(decoded_image_data, np.uint8), cv2.IMREAD_COLOR)
        shape = np.array(decoded_image.shape[:2], dtype=np.float32).reshape((1, 2))
        img = self.letterbox_image(decoded_image, self.input_size)
        img = img.astype(np.float32)
        img /= 255.0
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        data['task_info']['img'] = img
        data['task_info']['img_shape'] = shape
        return data

    def forward(self, data):
        output = self.session.run(output_names=self.output_names,
                                  input_feed={self.input_name: data['task_info']['img'],
                                              'image_shape': data['task_info']['img_shape']})
        data['task_info']['result'] = output
        del data['task_info']['img'], data['task_info']['img_shape']
        return data

    def postprocess(self, data):
        result = data['task_info']['result']
        boxes, scores, indices = result
        batch_size = boxes.shape[0]
        detection_result = [[] for _ in range(batch_size)]
        for i, c in reversed(list(enumerate(indices))):
            predicted_class = self.object_classes[c[1]]
            box = boxes[c[0]][c[2]].tolist()
            score = scores[c[0]][c[1]][c[2]].tolist()
            if score >= self.score_threshold:
                detection_result[c[0]].append({'score': score, 'class': predicted_class, "box": box})
        data['task_info']['result'] = detection_result
        return data

    def __call__(self, data):
        return self.postprocess(self.forward(self.preprocess(data)))


# Implement your handle here
@registry.reg()
class CustomHandle(BaseHandle):
    def __init__(self, gpu):
        pass

    def preprocess(self, data):
        pass

    def forward(self, data):
        pass

    def postprocess(self, data):
        pass

    def __call__(self, data):
        pass

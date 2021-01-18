# ml_serving_benchmark

### 1. 拉取Docker镜像

[TensorFlow Serving](https://github.com/tensorflow/serving)

```
docker pull tensorflow/serving
```

[AI-Serving](https://github.com/autodeployai/ai-serving)

```
docker pull autodeployai/ai-serving
```

### 2. 模型训练

- 将[数据文件](https://www.kaggle.com/leonerd/criteo-small)下载至data目录

- 启动训练

    ```bash
    mkdir outputs

    python train.py
    ```

### 3. 启动服务

- tf-serving

  ```bash
  docker run --rm -it \
    -p 8500:8500 -p 8501:8501 \
    -v ${PWD}/outputs/dcn:/models/dcn/1:ro -e MODEL_NAME=dcn \
    tensorflow/serving
  ```

- ai-serving

  1. 下载[原始proto文件](https://github.com/autodeployai/ai-serving/tree/master/src/main/protobuf)

  2. 生成protobuf和grpc的代码

      ```bash
      python -m grpc_tools.protoc \
        -I=./proto --python_out=. --grpc_python_out=. \
        ai-serving.proto onnx-ml.proto
      ```

  3. 将模型转换为ONNX格式

      ```bash
      python convert_onnx.py
      ```

  4. 启动服务

      ```bash
      mkdir outputs/ai-serving

      docker run --rm -it \
        -p 9090:9090 -p 9091:9091 \
        -v ${PWD}/outputs/ai-serving:/opt/ai-serving \
        autodeployai/ai-serving
      ```

  5. 发布变换函数

      ```bash
      curl -X PUT --data-binary @outputs/trans.onnx \
        -H "Content-Type: application/x-protobuf" \
        http://localhost:9090/v1/models/trans
      ```

  6. 发布模型

      ```bash
      curl -X PUT --data-binary @outputs/lgb.onnx \
        -H "Content-Type: application/x-protobuf" \
        http://localhost:9090/v1/models/lgb
      ```

- flask

  ```
  gunicorn flask_server:app
  ```

### 4. 启动性能测试

```bash
# 可更改为其他测试文件
TEST_FILE=tf_serving_http.py

locust -f ${TEST_FILE} --headless --run-time=30s -u=50 -r=1000
```

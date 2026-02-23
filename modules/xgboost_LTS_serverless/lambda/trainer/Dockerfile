# Dùng container image vì package size > 250MB (xgboost + numpy + pandas)
FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

# System deps cho xgboost (libgomp = OpenMP runtime)
RUN dnf install -y libgomp && dnf clean all

# Copy project source (modules cần cho training pipeline)
COPY modules/ ./modules/
COPY config/ ./config/
COPY requirements_trainer.txt ./

# Copy handler lên root để tránh 'lambda' Python keyword trong import path
# Lambda runtime dùng importlib nên path dotted không đi qua Python parser
# nhưng để đảm bảo tương thích, đặt handler ở root là best practice
COPY modules/xgboost_LTS_serverless/lambda/trainer/handler.py ./trainer_handler.py

RUN pip install --no-cache-dir -r requirements_trainer.txt

# CMD format: "module.function" — trainer_handler = tên file tại /var/task
CMD ["trainer_handler.handler"]

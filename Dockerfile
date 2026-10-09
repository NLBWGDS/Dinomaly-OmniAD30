FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DINOMALY_WEIGHTS_DIR=/opt/omniad/backbones/weights

WORKDIR /opt/omniad

COPY requirements-platform.txt ./
RUN python -m pip install --no-cache-dir -r requirements-platform.txt

COPY . ./
RUN mkdir -p /opt/omniad/backbones/weights && \
    python -c "import pathlib,urllib.request; p=pathlib.Path('/opt/omniad/backbones/weights/dinov2_vitb14_reg4_pretrain.pth'); urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth', p)" && \
    mkdir -p /home/apps && \
    ln -sfn /opt/omniad /home/apps/ats-reasoning-tool && \
    ln -sfn /opt/omniad/root/train.py /root/train.py && \
    sed -i 's/\r$//' platform_entrypoint.sh platform_launch.sh start.sh train.sh && \
    chmod +x platform_entrypoint.sh platform_launch.sh start.sh train.sh && \
    bash -n platform_launch.sh && \
    python -m py_compile platform_bootstrap.py platform_train.py platform_infer.py root/train.py && \
    python -c "import cv2,numpy,torch,timm,torchvision; print('platform dependencies OK', torch.__version__, torch.version.cuda)"

LABEL org.opencontainers.image.version="1.0.3"
CMD ["/bin/bash", "-c", "cd /home/apps/ats-reasoning-tool/ && sh start.sh /input/ /output/"]

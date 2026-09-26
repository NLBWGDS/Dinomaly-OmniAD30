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
    chmod +x /opt/omniad/platform_entrypoint.sh /opt/omniad/start.sh /opt/omniad/train.sh && \
    python -c "import cv2,numpy,torch,timm,torchvision; print('platform dependencies OK', torch.__version__, torch.version.cuda)"

CMD ["/bin/bash", "-c", "cd /opt/omniad && sh start.sh /input/ /output/"]

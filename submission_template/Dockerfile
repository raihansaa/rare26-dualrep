FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS example-algorithm-amd64
# Pinned, not ':latest' -- that tag is stale at torch 2.2.1, whose kernels stop at
# sm_90 and cannot run on Blackwell, making local verification impossible. This tag
# matches the torch that wrote the checkpoints (2.11.0+cu128). 'runtime' not 'devel':
# 4.3 GB against 14.1 GB, and nothing here compiles.

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user model /opt/app/model

# You can add any Python dependencies to requirements.txt
# --break-system-packages: this base uses Debian's system python, which carries a
# PEP 668 marker that makes pip refuse to install at all. Nothing is actually at
# risk -- --user installs into ~/.local, and requirements.txt deliberately holds no
# package that the base image already provides.
RUN python -m pip install \
    --user \
    --break-system-packages \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# platt.json carries one (a, b) per checkpoint, each fitted on that model's own
# out-of-fold predictions at build time. inference.py raises if any is missing,
# so a forgotten copy fails loudly rather than shipping uncalibrated scores.
COPY --chown=user:user platt.json /opt/app/
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]

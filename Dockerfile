# Used to debug codspeed locally without having to upload to CI
# docker build -t egglog .
# docker run --rm -v $PWD:/code/ egglog

# codspeed requires ubuntu
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    lsb-release \
    build-essential \
    curl \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/rust-lang/docker-rust/blob/master/Dockerfile-debian.template
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.71.1 --no-modify-path  --profile minimal
RUN ls /usr/local/cargo

RUN curl -fsSL https://github.com/CodSpeedHQ/runner/releases/download/v2.1.1/codspeed-runner-installer.sh | bash


WORKDIR /code/
RUN python3.12 -m venv .venv
ENV PATH=/code/.venv/bin:$PATH


COPY src /code/src
COPY README.md Cargo.toml Cargo.lock pyproject.toml conftest.py rust-toolchain /code/
RUN pip install -e .[test]


# https://github.com/CodSpeedHQ/runner/blob/93f634af1d5ceae2c36a240c6be2dc8601b82220/src/ci_provider/buildkite/provider.rs#L197
ENV BUILDKITE_REPO=https://github.com/egraphs-good/egglog-python.git
ENV BUILDKITE_BRANCH=_LOCAL_
ENV BUILDKITE_COMMIT=_LOCAL_
ENV BUILDKITE_AGENT_NAME=_LOCAL_
ENV BUILDKITE_ORGANIZATION_SLUG=_LOCAL_
ENV BUILDKITE_PIPELINE_SLUG=_LOCAL_
ENV BUILDKITE_PULL_REQUEST_BASE_BRANCH=
ENV BUILDKITE_PULL_REQUEST=
ENV BUILDKITE=true
CMD codspeed-runner --token=ed77fc2b-c06a-4457-8d5a-c6e43077b6f6  -- pytest -vvv --codspeed

FROM ubuntu:22.04

# Аргументы для UID/GID
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=oversir

# Настройка прокси
ENV HTTPS_PROXY=http://user334497:39yesv@216.74.104.140:4751
ENV HTTP_PROXY=http://user334497:39yesv@216.74.104.140:4751

# Установка базовых зависимостей и очистка кеша
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    procps \
    file \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя с указанными UID/GID
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Переключение на пользователя для установки Homebrew
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Установка Homebrew
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && \
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/${USERNAME}/.bashrc

# Добавление Homebrew в PATH
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"

# Установка Claude Code через Homebrew
RUN eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" && \
    brew install claude-code

# Установка uv (менеджер Python окружений)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /home/${USERNAME}/.bashrc

ENV PATH="/home/${USERNAME}/.cargo/bin:${PATH}"

# Настройка для NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

CMD ["/bin/bash"]
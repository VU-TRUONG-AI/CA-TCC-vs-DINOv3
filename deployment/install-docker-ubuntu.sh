#!/usr/bin/env bash
# Official Docker apt repository; run interactively on the Ubuntu host.
set -euo pipefail
. /etc/os-release
if [[ "$ID" != ubuntu ]]; then
  echo 'This installer is for Ubuntu only.' >&2
  exit 1
fi
if command -v docker >/dev/null; then
  echo 'Docker is already installed; keeping the current installation.'
  docker --version
  exit 0
fi
sudo -v
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
TASK_DOCKER_ARCH=$(dpkg --print-architecture)
printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' "${UBUNTU_CODENAME:-$VERSION_CODENAME}" "$TASK_DOCKER_ARCH" | sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
printf '\nDocker installed successfully. Run Docker commands with sudo.\n'

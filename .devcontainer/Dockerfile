FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm
# The base image ships a yarn apt source whose GPG key has expired (NO_PUBKEY 62D54FD4003F6525),
# which breaks apt-get update and therefore every feature that installs packages. Python work needs no yarn.
RUN rm -f /etc/apt/sources.list.d/yarn.list

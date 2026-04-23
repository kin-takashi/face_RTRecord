#!/bin/bash
set -e

# Create necessary dirs (backup for config.py)
mkdir -p dataset embeddings logs predata videos

# Exec the command passed to docker run
exec "$@"


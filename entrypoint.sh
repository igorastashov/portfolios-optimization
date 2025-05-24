#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Execute the command passed into this script (i.e., the CMD from Dockerfile or command from docker-compose.yml)
echo "Executing command: $@"
exec "$@" 
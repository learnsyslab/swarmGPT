#!/usr/bin/env bash

# Check if the node_modules directory exists inside the 'web' folder
if [ ! -d "web/node_modules" ]; then
  echo "Frontend dependencies not found. Installing..."
  npm --prefix web install
else
  echo "Frontend dependencies already installed. Skipping..."
fi
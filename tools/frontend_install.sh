#!/usr/bin/env bash

main() {
  local package_json="web/package.json"
  local package_lock="web/package-lock.json"
  local node_modules="web/node_modules"

  if [ ! -f "$package_json" ]; then
    echo "Frontend package web/package.json not found. Skipping frontend dependency install."
    return 0
  fi

  if [ ! -d "$node_modules" ] \
    || [ "$package_json" -nt "$node_modules" ] \
    || { [ -f "$package_lock" ] && [ "$package_lock" -nt "$node_modules" ]; }; then
    echo "Frontend dependencies missing or stale. Installing..."
    npm --prefix web install
  else
    echo "Frontend dependencies already installed and up to date. Skipping..."
  fi
}

main "$@"

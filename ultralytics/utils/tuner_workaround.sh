#!/bin/bash

# Define the path to the virtual environment's activate script
venv_activate_script="/home-net/ierregue/.virtualenvs/small-fast-detector/bin/activate"  # Replace with your actual path

# Check if the virtual environment activate script exists
if [ ! -f "$venv_activate_script" ]; then
    echo "Virtual environment activation script not found."
    exit 1
fi

# Activate the virtual environment
source "$venv_activate_script"

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <command>"
  exit 1
fi

# Join all the arguments into a single string
command="$*"

# Use eval to execute the command
eval "$command"


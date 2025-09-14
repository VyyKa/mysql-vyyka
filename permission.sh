#!/bin/bash

# Set this to your logs directory
LOG_DIR="./logs"

# Create folder if not exists
mkdir -p "$LOG_DIR"

# Create log files if missing
touch "$LOG_DIR/mysql.log"
touch "$LOG_DIR/alerts.log"

# Set permissions
chmod 755 "$LOG_DIR"           # Allow containers to access folder
chmod 666 "$LOG_DIR/mysql.log" # Read/write for all users (MySQL writes, Logstash/AI reads)
chmod 666 "$LOG_DIR/alerts.log"

echo "✅ Log permissions set:"
ls -l "$LOG_DIR"

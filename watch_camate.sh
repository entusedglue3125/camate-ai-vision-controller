#!/usr/bin/env bash
# Camate Camera Live Feed Launcher
# Camera IP: 10.46.187.6

MAIN_STREAM="rtsp://admin:@10.46.187.6:554/ch0_0.264"
SUB_STREAM="rtsp://admin:@10.46.187.6:554/ch0_1.264"

echo "=============================================="
echo " Starting Camate Camera Live Stream Viewer    "
echo "=============================================="
echo "Main Stream: $MAIN_STREAM (2304x1296 2K HEVC)"
echo "Sub  Stream: $SUB_STREAM (800x448)"
echo "----------------------------------------------"

# Launch VLC with 300ms network caching
vlc --network-caching=300 "$MAIN_STREAM" "$@"

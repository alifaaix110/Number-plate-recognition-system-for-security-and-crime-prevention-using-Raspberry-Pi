#!/bin/bash

echo "Updating system"
sudo apt update

echo "Installing system packages"
sudo apt install -y python3-pip python3-opencv tesseract-ocr

echo "Installing python packages"
pip3 install -r requirements.txt

echo "Setup finished"

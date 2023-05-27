#!/bin/bash
echo "Updating image!"
sudo docker rmi proj:img
sudo docker build -f all.Dockerfile -t proj:img .

#!/bin/bash
echo "Updating image!"
docker rmi proj:img
docker build -f all.Dockerfile -t proj:img .

#!/bin/bash

docker build -t logo_matcher . && docker image prune -f

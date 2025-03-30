#!/bin/bash

touch clusters.json
docker run --rm -v $(pwd)/visualizations:/visualizations -v $(pwd)/clusters.json:/clusters.json logo_matcher

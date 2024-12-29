# occupancy-networks

![Project Logo](link-to-your-logo.png) <!-- Replace with your project logo if available -->

## Overview

**Occupancy Networks** predict 3D occupancy of surrounding environments, given multi-view camera images as input. This repository contains the rapid prototype of implementations inspired by Tesla's occupancy networks architecture outlined in [Tesla AI Day 2022](https://www.youtube.com/live/ODSJsviD_SU?si=8Moz17sixqAEIyiX). 

## Features

- **TBD**: 

## Installation

To get started with Occupancy Networks, ensure you have anaconda installed. Then follow these steps:

1. **Clone the repository**:
```bash
git clone https://github.com/obikata/occupancy-networks.git
cd occupancy-networks
```

2. **Set up the environment (assuming use of Anaconda for environment management)**:
```bash
conda env create -f environment.yaml
conda activate occnet
```

## Demo

### Blender Stereo Images

#### Train
```python
python train.py
```

#### Test
```python
python test.py
```

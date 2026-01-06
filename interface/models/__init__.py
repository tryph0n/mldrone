"""
Model definitions for DroneDetect.
"""
from .base import BaseModel
from .cnn import VGG16FC, ResNet50FC, CNNModelAdapter
from .rfuavnet import RFUAVNet, RFUAVNetAdapter
from .svm import SVMAdapter

__all__ = [
    'BaseModel',
    'VGG16FC', 'ResNet50FC', 'CNNModelAdapter',
    'RFUAVNet', 'RFUAVNetAdapter',
    'SVMAdapter',
]

#!/usr/bin/env python3
"""
AI Dataset Labeler
A comprehensive tool for creating training datasets for AI models
Supports YOLOv8, MoveNet, MoViNet, and image generator formats
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ai_labeler import AILabelerMainWindow

def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("AI Dataset Labeler")
    app.setApplicationVersion("1.0.0")
    
    # Create main window
    window = AILabelerMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
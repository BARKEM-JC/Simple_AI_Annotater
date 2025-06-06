@echo off
echo Building AI Labeler Executable (Minimal Build)...
echo.

REM Install minimal dependencies for executable build
echo Installing minimal dependencies for build...
pip install -r requirements.txt
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Build with minimal dependencies
echo Building executable with minimal dependencies...
python -m PyInstaller --clean --onedir --windowed ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module tensorflow ^
    --exclude-module keras ^
    --exclude-module matplotlib ^
    --exclude-module pandas ^
    --exclude-module scipy ^
    --exclude-module sklearn ^
    --exclude-module sympy ^
    --exclude-module jupyter ^
    --exclude-module IPython ^
    --exclude-module notebook ^
    --exclude-module seaborn ^
    --exclude-module plotly ^
    --exclude-module bokeh ^
    --exclude-module dash ^
    --exclude-module streamlit ^
    --exclude-module flask ^
    --exclude-module django ^
    --exclude-module fastapi ^
    --exclude-module sqlalchemy ^
    --exclude-module psutil ^
    --exclude-module h5py ^
    --exclude-module tables ^
    --exclude-module numba ^
    --exclude-module dask ^
    --exclude-module multiprocessing ^
    --exclude-module tkinter ^
    --exclude-module unittest ^
    --exclude-module doctest ^
    --add-data "custom_labels.json;." ^
    --add-data "opencv_templates.json;." ^
    --name AILabeler ^
    main.py

if exist "dist\AILabeler\AILabeler.exe" (
    echo.
    echo ===================================
    echo Minimal build completed successfully!
    echo Executable location: dist\AILabeler\AILabeler.exe
    echo Folder location: dist\AILabeler\
    echo ===================================
    echo.
    echo You can now distribute the entire 'dist\AILabeler' folder.
    echo.
    pause
) else (
    echo.
    echo ===================================
    echo Build failed! Check the output above for errors.
    echo ===================================
    echo.
    pause
) 
@echo off
echo Building AI Labeler Executable...
echo.

REM Install minimal dependencies for executable build
echo Installing minimal dependencies for build...
pip install -r requirements.txt
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Always use our custom spec file
echo Using custom spec file for minimal build...

REM Build the executable
echo Building executable (this may take several minutes)...
python -m PyInstaller --clean AILabeler.spec

if exist "dist\AILabeler\AILabeler.exe" (
    echo.
    echo ===================================
    echo Build completed successfully!
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
    echo Common solutions:
    echo 1. Try installing visual studio build tools
    echo 2. Use a virtual environment
    echo 3. Check if antivirus is blocking the build
    echo.
    pause
) 
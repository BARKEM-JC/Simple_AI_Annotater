# Building AI Labeler Executable

This document provides instructions for creating a standalone executable of the AI Labeler application.

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Quick Build (Recommended)

### Windows
1. Open Command Prompt or PowerShell in the project directory
2. Run the build script:
   ```cmd
   .\build_executable.bat
   ```

### Linux/Mac
1. Open Terminal in the project directory
2. Make the script executable (if not already):
   ```bash
   chmod +x build_executable.sh
   ```
3. Run the build script:
   ```bash
   ./build_executable.sh
   ```

## Manual Build Process

If you prefer to build manually or need to customize the build:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Clean Previous Builds (Optional)
```bash
# Windows
rmdir /s build dist

# Linux/Mac
rm -rf build dist
```

### Step 3: Build the Executable
```bash
python -m PyInstaller AILabeler.spec
```

## Output

The executable will be created in the `dist` folder:
- **Windows:** `dist/AILabeler.exe`
- **Linux/Mac:** `dist/AILabeler`

## Customization Options

### Adding an Icon
1. Place your `.ico` file (Windows) or `.icns` file (Mac) in the project directory
2. Edit `AILabeler.spec` and update the `icon` parameter:
   ```python
   icon='your_icon.ico'  # Windows
   icon='your_icon.icns'  # Mac
   ```

### Including Additional Files
To include additional data files in the executable, edit the `datas` list in `AILabeler.spec`:
```python
datas=[
    ('custom_labels.json', '.'),
    ('your_file.txt', '.'),
    ('folder_name', 'folder_name'),
],
```

### Debug Mode
For troubleshooting, you can enable debug mode by editing `AILabeler.spec`:
```python
debug=True,
console=True,  # Shows console window for debugging
```

## Troubleshooting

### Common Issues

1. **Missing Module Errors:**
   - Add missing modules to the `hiddenimports` list in `AILabeler.spec`

2. **Large Executable Size:**
   - The executable includes all dependencies and may be 200-500MB
   - This is normal for PyQt5 applications with OpenCV

3. **Slow Startup:**
   - First run may be slower as the executable extracts files
   - Subsequent runs should be faster

4. **PyQt5 Import Errors:**
   - Ensure PyQt5 is properly installed: `pip install PyQt5`
   - Try reinstalling: `pip uninstall PyQt5 && pip install PyQt5`

5. **OpenCV Issues:**
   - If opencv-python causes issues, try: `pip install opencv-python-headless`

6. **PyInstaller Not Found:**
   - Use `python -m PyInstaller` instead of just `pyinstaller`
   - Ensure PyInstaller is installed: `pip install pyinstaller`

### Building for Distribution

For distributing the executable:

1. **Windows:**
   - The `.exe` file in `dist` folder is standalone
   - Users don't need Python installed
   - Include any additional data files alongside the executable

2. **Linux:**
   - The executable may require certain system libraries
   - Test on the target distribution
   - Consider using `--onedir` instead of `--onefile` for better compatibility

3. **Mac:**
   - The executable should work on systems with the same or newer macOS version
   - For broader compatibility, build on an older macOS version

## Performance Notes

- The executable will be larger than the source code (typically 200-500MB)
- Startup time may be 2-5 seconds on first run
- Runtime performance should be similar to running the Python script directly

## Alternative Build Tools

If PyInstaller doesn't work for your use case, consider:
- **cx_Freeze:** `pip install cx_Freeze`
- **auto-py-to-exe:** GUI wrapper for PyInstaller
- **Nuitka:** For potentially better performance 
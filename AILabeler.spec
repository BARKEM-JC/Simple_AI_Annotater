# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('custom_labels.json', '.'), ('opencv_templates.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras', 'matplotlib', 'pandas', 'scipy', 'sklearn', 'sympy', 'jupyter', 'IPython', 'notebook', 'seaborn', 'plotly', 'bokeh', 'dash', 'streamlit', 'flask', 'django', 'fastapi', 'sqlalchemy', 'psutil', 'h5py', 'tables', 'numba', 'dask', 'multiprocessing', 'tkinter', 'unittest', 'doctest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AILabeler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AILabeler',
)

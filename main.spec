# -*- mode: python ; coding: utf-8 -*-


from PyInstaller.utils.hooks import collect_data_files
import ultralytics
ultra_files = collect_data_files('ultralytics')

block_cipher = None

a = Analysis(
    ['test_model.py'], # <- Please change this into your python code name.
    pathex=['.'],
    binaries=[],
    datas=ultra_files + [
    ('license_car_noV2.pt', '.'),
    ('licensr_position.pt', '.'),
    ('car_no.pt', '.')], # <- This is for enabling the referencing of all data files in the Ultralytics library.
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='test_model',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='test_model',
)
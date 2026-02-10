# Fix for ldecod.exe Runtime Dependencies

## Problem
The `ldecod.exe` compiled in the GitHub Actions workflow failed to run on Windows systems because it was missing required runtime DLL dependencies:
- `libpng16-16.dll` (PNG image library)
- `libtiff-6.dll` (TIFF image library)
- `libjpeg-8.dll` (JPEG image library)
- `libwinpthread-1.dll` (POSIX threading library for Windows)

## Root Cause
The workflow was compiling `ldecod.exe` using MSYS2/MinGW with dynamic linking to these libraries. While the development packages were installed during the build process (for headers and link-time libraries), the runtime DLL files were not being copied into the distribution package. This meant that when the executable was run on a system without MSYS2, it couldn't find these DLLs.

## Solution
Modified `.github/workflows/package.yml` to:

1. **Copy Core Runtime DLLs**: After compiling `ldecod.exe`, the workflow now copies the essential runtime DLLs from MSYS2/MinGW to the `dist_bin` directory:
   - `libpng16-16.dll`
   - `libtiff-6.dll`
   - `libjpeg-8.dll`
   - `libwinpthread-1.dll`

2. **Copy Transitive Dependencies**: Also copies additional DLLs that may be required by the above libraries (using `|| true` to avoid failures if some are not present):
   - `zlib1.dll` (compression, used by PNG and TIFF)
   - `libdeflate.dll` (compression, used by TIFF)
   - `libjbig-0.dll` (JBIG compression for TIFF)
   - `libLerc.dll` (LERC compression for TIFF)
   - `liblzma-5.dll` (LZMA compression for TIFF)
   - `libzstd.dll` (Zstandard compression for TIFF)
   - `libsharpyuv-0.dll` (YUV conversion)
   - `libwebp-7.dll` (WebP format support)

3. **Bundle with PyInstaller**: These DLLs in `dist_bin` are automatically included in the final PyInstaller package via the existing `--add-data "dist_bin/*;dist_bin/"` flag.

## Changes Made
- Added DLL copying commands to the "Compile JM Decoder" step in `.github/workflows/package.yml`
- Updated workflow trigger to include the `copilot/fix-ldecod-runtime-dependencies` branch for testing

## Testing
The workflow will run automatically on push to the specified branches. The packaged executables (`video_analysis_cmd.exe` and `video_analysis_gui.exe`) will now include all necessary DLLs to run ldecod.exe without requiring MSYS2 installation.

## Future Improvements
Consider using static linking (`STC=1` in the Makefile) to avoid DLL dependencies entirely, though this may increase the executable size.

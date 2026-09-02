# cibuildwheel before-all for Windows. See install-deps-almalinux.sh: COLMAP is
# built in-tree by FetchContent, so this only provides ccache and vcpkg.
# MSVC provides its own OpenMP runtime, so no libomp equivalent is needed.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $true

$CURRDIR = $PWD

$COMPILER_TOOLS_DIR = "${env:COMPILER_CACHE_DIR}/bin"
New-Item -ItemType Directory -Force -Path ${COMPILER_TOOLS_DIR}
$env:Path = "${COMPILER_TOOLS_DIR};" + $env:Path

If (!(Test-Path -path "${COMPILER_TOOLS_DIR}/ccache.exe" -PathType Leaf)) {
    misc/ci/install-ccache.ps1 -Destination "${COMPILER_TOOLS_DIR}"
}
ccache --zero-stats

cd ${CURRDIR}
git clone https://github.com/microsoft/vcpkg ${env:VCPKG_INSTALLATION_ROOT}
cd ${env:VCPKG_INSTALLATION_ROOT}
./bootstrap-vcpkg.bat

cd ${CURRDIR}
& "./misc/ci/enter_vs_dev_shell.ps1"
& "${env:VCPKG_INSTALLATION_ROOT}/vcpkg.exe" integrate install

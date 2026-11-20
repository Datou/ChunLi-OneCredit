# Build a browser-loadable Genesis Plus GX core that EXPORTS the libretro memory API.
#
# Why this exists
# ---------------
# EmulatorJS ships a Genesis Plus GX build whose 67 WASM exports are minified
# (Ch, Dh, ...) and which does NOT export retro_get_memory_data, so a browser
# frontend cannot read 68K work RAM.  The chunppo action table resolves FWD/BACK
# from RAM (P1_X=0x8006 / P2_X=0x8286), so without RAM the policy cannot be
# driven correctly.
#
# Upstream Genesis Plus GX (github.com/libretro/Genesis-Plus-GX) builds the full
# libretro ABI, but its `platform=emscripten` target only produces LLVM bitcode
# (an object archive with a .bc suffix). Linking that archive ourselves is
# exactly where we choose the export list -- and we add retro_get_memory_data.
#
# Prerequisites (prepare once):
#   * libretro/Genesis-Plus-GX checked out at b9c2316441978d4015ba30e9a95d9b0e6f8ee981
#     or openai/retro v0.8.0's cores/genesis with -GymRetro (current web build)
#   * make on PATH            (conda: `conda install -c conda-forge make`)
#   * emcc on PATH            (emsdk: `emsdk install latest && emsdk activate latest`)
#
# License: Genesis Plus GX is source-available under a NON-COMMERCIAL license
# (see LICENSE.txt).  Building a modified core for personal use is fine; do not
# redistribute the built binary commercially.

[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GxRoot,
    [string]$OutDir   = (Join-Path $PSScriptRoot '../web/cores'),
    [int]$Jobs        = 8,
    [switch]$GymRetro
)

$ErrorActionPreference = 'Stop'

function Require-Command([string]$name, [string]$hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "missing '$name' on PATH. $hint"
    }
}

Require-Command make "install GNU make (conda install -c conda-forge make)"
Require-Command emcc "install Emscripten (emsdk install latest; emsdk activate latest)"
Require-Command emar "activate the Emscripten SDK so emar is also on PATH"

if (-not (Test-Path "$GxRoot\Makefile.libretro")) { throw "Genesis Plus GX not found at $GxRoot" }
if (-not (Test-Path "$GxRoot\libretro\libretro-common\include")) {
    throw "missing libretro-common headers in $GxRoot"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

# Standard libretro frontend ABI, plus the memory accessors that this whole
# exercise exists for.  malloc/free are needed to hand the ROM to the core.
$exports = @(
    '_retro_init', '_retro_deinit', '_retro_api_version',
    '_retro_get_system_info', '_retro_get_system_av_info',
    '_retro_set_environment', '_retro_set_video_refresh',
    '_retro_set_audio_sample', '_retro_set_audio_sample_batch',
    '_retro_set_input_poll', '_retro_set_input_state',
    '_retro_set_controller_port_device',
    '_retro_reset', '_retro_run',
    '_retro_serialize_size', '_retro_serialize', '_retro_unserialize',
    '_retro_cheat_reset', '_retro_cheat_set',
    '_retro_load_game', '_retro_unload_game', '_retro_get_region',
    '_retro_get_memory_data', '_retro_get_memory_size',
    '_malloc', '_free'
) -join ','

Write-Host "== 1/2 compiling core to LLVM bitcode ==" -ForegroundColor Cyan
Push-Location $GxRoot
$archive = $null
try {
    $makeFlags = @('CC=emcc', 'CXX=em++', 'AR=emar')
    if ($GymRetro) {
        # Gym Retro v0.8.0's cores/genesis predates Clang's -fno-common default.
        $makeFlags = @('CC=emcc -fcommon', 'CXX=em++', 'AR=emar', 'HAVE_CHD=0')
    }
    & make -f Makefile.libretro platform=emscripten @makeFlags -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "make failed with exit code $LASTEXITCODE" }

    $bc = Get-ChildItem -Path $GxRoot -Filter '*_libretro_emscripten.bc' -Recurse |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $bc) { throw 'no *_libretro_emscripten.bc produced' }
    Write-Host "bitcode: $($bc.FullName)  ($([math]::Round($bc.Length/1MB,1)) MB)"

    Write-Host "== 2/2 linking standalone module with exported memory API ==" -ForegroundColor Cyan
    $outJs = Join-Path $OutDir 'genesis_plus_gx.js'
    # A .bc suffix makes newer emcc try to parse the archive as LLVM IR.
    $archive = Join-Path ([System.IO.Path]::GetTempPath()) ("genesis-plus-gx-" + [guid]::NewGuid() + ".a")
    Copy-Item -LiteralPath $bc.FullName -Destination $archive
    # The static core omits the libretro-common file system implementations.
    $common = 'libretro/libretro-common'
    $support = @(
        'streams/file_stream.c', 'streams/file_stream_transforms.c',
        'compat/fopen_utf8.c', 'compat/compat_snprintf.c', 'compat/compat_strl.c',
        'compat/compat_strcasestr.c', 'compat/compat_posix_string.c',
        'encodings/encoding_utf.c', 'file/file_path.c', 'file/retro_dirent.c',
        'lists/string_list.c', 'lists/dir_list.c', 'memmap/memalign.c',
        'string/stdstring.c', 'vfs/vfs_implementation.c'
    ) | ForEach-Object { "$common/$_" }
    $ports = @()
    if ($GymRetro) {
        $support = $support | Where-Object { $_ -match '/streams/' }
        $ports = @('-s', 'USE_ZLIB=1')
    }
    & emcc $archive @support @ports "-I$common/include" -O3 --no-entry -o $outJs `
        -s WASM=1 `
        -s MODULARIZE=1 `
        -s EXPORT_NAME=createGenesisPlusGX `
        -s ALLOW_MEMORY_GROWTH=1 `
        -s INITIAL_MEMORY=67108864 `
        -s FORCE_FILESYSTEM=1 `
        -s ALLOW_TABLE_GROWTH=1 `
        -s EXPORTED_FUNCTIONS="[$exports]" `
        -s EXPORTED_RUNTIME_METHODS='["FS","addFunction","stringToUTF8","HEAPU8","HEAP32"]' `
        -s ASSERTIONS=0
    if ($LASTEXITCODE -ne 0) { throw "emcc link failed with exit code $LASTEXITCODE" }

    Write-Host "`n== built ==" -ForegroundColor Green
    Get-ChildItem $OutDir -Filter 'genesis_plus_gx.*' |
        ForEach-Object { '  {0}  {1:N1} MB' -f $_.Name, ($_.Length / 1MB) }
}
finally {
    if ($archive -and (Test-Path -LiteralPath $archive)) { Remove-Item -LiteralPath $archive }
    Pop-Location
}

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

if [[ ! -d "${TP}/FARM-Project/.git" ]]; then
  git clone --recursive https://github.com/GoldenGait/FARM-Project.git "${TP}/FARM-Project"
else
  echo "FARM-Project already present"
fi

if [[ ! -d "${TP}/ss-3dgs/.git" ]]; then
  git clone https://github.com/Kodifly/ss-3dgs.git "${TP}/ss-3dgs"
else
  echo "ss-3dgs already present"
fi

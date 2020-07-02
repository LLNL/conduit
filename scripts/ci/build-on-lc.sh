#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color

# Update uberenv if variable set
if [[ -n "${UPDATE_UBERENV}" ]]
then
    echo "${RED}[!!!] This pipeline tests the update of uberenv!${NC}"
    ./scripts/ci/update-uberenv.sh "${UPDATE_UBERENV}"
fi

# Switch between dependency and install modes
INSTALL=""
if [[ "${UBERENV_INSTALL}" == "ON" ]]
then
    INSTALL="--install --run_tests"
fi

mkdir -p ${INSTALL_ROOT}_${SYS_TYPE}_${TOOLCHAIN}

# Preparing command
PREFIX="--prefix=${INSTALL_ROOT}_${SYS_TYPE}_${TOOLCHAIN}"
SPEC="--spec=${PKG_SPEC}"
MIRROR="--mirror=${SPACK_MIRROR}"
UBERENV_CMD="scripts/uberenv/uberenv.py ${INSTALL} ${SPEC} ${PREFIX} ${MIRROR}"

# Build command
echo "[run] ${UBERENV_CMD}"
${UBERENV_CMD}

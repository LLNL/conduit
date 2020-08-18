#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color

# Update uberenv if variable set
if [[ -n "${UPDATE_UBERENV}" ]]
then
    echo "${RED}[!!!] This pipeline tests the update of uberenv!${NC}"
    ./scripts/ci/update-uberenv.sh "${UPDATE_UBERENV}"
fi

# Change spack config to use a maximum of 64 threads for build.
pattern="^  build_jobs: 8$"
config_file="scripts/uberenv/spack_configs/config.yaml"

if grep -q -e "${pattern}" ${config_file}
then
    echo "Setting spack to build with maximum 64 threads"
    # Note: the following command isn’t compatible with MacOS,
    # where the equivalent would be to change "-i" with "-i ''".
    sed -i "s/${pattern}/  build_jobs: 64/" ${config_file}
else
    echo "ERROR: build_jobs setting in spack config has changed."
    echo "Please update ${0} accordingly".
    exit 1
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

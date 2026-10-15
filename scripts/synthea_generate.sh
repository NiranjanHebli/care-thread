#!/usr/bin/env bash
# Synthea synthetic patient generator script
# Seeded, reproducible CSV generation for Massachusetts population
set -euo pipefail

SEED="${1:-42}"
PATIENTS="${2:-100}"
STATE="${3:-Massachusetts}"
OUTPUT_DIR="${4:-data/synthea/csv}"
JAR_PATH="data/synthea/synthea-with-dependencies.jar"
SYNTHEA_RELEASE_URL="https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${JAR_PATH}")"

echo "=========================================================="
echo "CareThread Synthea Generator"
echo "Seed:       ${SEED}"
echo "Patients:   ${PATIENTS}"
echo "State:      ${STATE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================================="

# Check if synthea jar already exists locally
if [ ! -f "${JAR_PATH}" ] && [ ! -f "./synthea-with-dependencies.jar" ]; then
    echo "Downloading Synthea generator jar from GitHub releases..."
    curl -L -o "${JAR_PATH}" "${SYNTHEA_RELEASE_URL}"
fi

ACTUAL_JAR="${JAR_PATH}"
if [ -f "./synthea-with-dependencies.jar" ]; then
    ACTUAL_JAR="./synthea-with-dependencies.jar"
fi

# Execute via host Java if available, otherwise via eclipse-temurin Docker container
if java -version >/dev/null 2>&1; then
    echo "Running Synthea via host Java..."
    java -jar "${ACTUAL_JAR}" \
        -s "${SEED}" -p "${PATIENTS}" \
        --exporter.csv.export=true \
        --exporter.fhir.export=false \
        --exporter.baseDirectory="${OUTPUT_DIR}" \
        "${STATE}"
elif command -v docker >/dev/null 2>&1; then
    echo "Running Synthea via eclipse-temurin Docker container..."
    docker run --rm \
        -v "$(pwd):/workspace" \
        -w /workspace \
        eclipse-temurin:17-jre \
        java -jar "${ACTUAL_JAR}" \
        -s "${SEED}" -p "${PATIENTS}" \
        --exporter.csv.export=true \
        --exporter.fhir.export=false \
        --exporter.baseDirectory="${OUTPUT_DIR}" \
        "${STATE}"
else
    echo "Error: Neither Java runtime nor Docker is available."
    exit 1
fi

echo "Synthea generation completed successfully in ${OUTPUT_DIR}."

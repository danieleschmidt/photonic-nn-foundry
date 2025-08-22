#!/bin/bash
set -e

# Rollback Script for Photonic Neural Network Foundry

NAMESPACE="photonic-foundry"
DEPLOYMENT_NAME="photonic-foundry-app"

echo "🔄 Starting rollback process..."

# Get current revision
CURRENT_REVISION=$(kubectl rollout history deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE} --revision=0 | tail -1 | awk '{print $1}')
echo "Current revision: ${CURRENT_REVISION}"

# Get previous revision
PREVIOUS_REVISION=$((CURRENT_REVISION - 1))
echo "Rolling back to revision: ${PREVIOUS_REVISION}"

# Perform rollback
kubectl rollout undo deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE} --to-revision=${PREVIOUS_REVISION}

# Wait for rollback to complete
kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE} --timeout=300s

echo "✅ Rollback completed successfully!"

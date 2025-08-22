#!/bin/bash

# Health Check Script for Photonic Neural Network Foundry

NAMESPACE="photonic-foundry"
SERVICE_NAME="photonic-foundry-service"

echo "🏥 Running production health checks..."

# Check pod status
echo "Checking pod status..."
kubectl get pods -n ${NAMESPACE} -l app=photonic-foundry

# Check service status
echo "Checking service status..."
kubectl get service ${SERVICE_NAME} -n ${NAMESPACE}

# Check HPA status
echo "Checking HPA status..."
kubectl get hpa -n ${NAMESPACE}

# Check ingress status
echo "Checking ingress status..."
kubectl get ingress -n ${NAMESPACE}

# Test health endpoint
echo "Testing health endpoint..."
kubectl port-forward service/${SERVICE_NAME} 8080:80 -n ${NAMESPACE} &
PORT_FORWARD_PID=$!

sleep 3

if curl -f http://localhost:8080/health; then
    echo "✅ Health endpoint responding"
else
    echo "❌ Health endpoint not responding"
fi

kill $PORT_FORWARD_PID

echo "🏥 Health check completed"

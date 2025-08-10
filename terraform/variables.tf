# Global Variables for Photonic Foundry Multi-Cloud Deployment

# Project Configuration
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
  default     = "photonic-foundry.com"
}

# Cloud Provider Enablement
variable "enable_aws" {
  description = "Enable AWS infrastructure deployment"
  type        = bool
  default     = true
}

variable "enable_gcp" {
  description = "Enable Google Cloud Platform infrastructure deployment"
  type        = bool
  default     = true
}

variable "enable_azure" {
  description = "Enable Azure infrastructure deployment"
  type        = bool
  default     = true
}

variable "enable_alibaba" {
  description = "Enable Alibaba Cloud infrastructure deployment"
  type        = bool
  default     = true
}

variable "enable_cloudflare" {
  description = "Enable Cloudflare for global DNS and CDN"
  type        = bool
  default     = true
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version to deploy across all clouds"
  type        = string
  default     = "1.28"
}

variable "enable_quantum_compute" {
  description = "Enable quantum compute node pools"
  type        = bool
  default     = true
}

# AWS Specific Variables
variable "aws_vpc_cidr_blocks" {
  description = "CIDR blocks for AWS VPCs by region"
  type = map(string)
  default = {
    "us-east-1"      = "10.1.0.0/16"
    "eu-west-1"      = "10.2.0.0/16"
    "ap-southeast-1" = "10.3.0.0/16"
  }
}

variable "aws_node_instance_types" {
  description = "EC2 instance types for AWS Kubernetes nodes"
  type = map(object({
    general_compute = list(string)
    quantum_compute = list(string)
    gpu_compute    = list(string)
  }))
  default = {
    "us-east-1" = {
      general_compute = ["m5.xlarge", "m5.2xlarge", "m5.4xlarge"]
      quantum_compute = ["c5n.4xlarge", "c5n.9xlarge", "c5n.18xlarge"]
      gpu_compute    = ["p4d.xlarge", "p4d.2xlarge", "p4d.8xlarge"]
    }
    "eu-west-1" = {
      general_compute = ["m5.xlarge", "m5.2xlarge", "m5.4xlarge"]
      quantum_compute = ["c5n.4xlarge", "c5n.9xlarge", "c5n.18xlarge"]
      gpu_compute    = ["p3.2xlarge", "p3.8xlarge", "p3dn.24xlarge"]
    }
    "ap-southeast-1" = {
      general_compute = ["m5.large", "m5.xlarge", "m5.2xlarge"]
      quantum_compute = ["c5n.2xlarge", "c5n.4xlarge", "c5n.9xlarge"]
      gpu_compute    = ["p3.2xlarge", "p3.8xlarge"]
    }
  }
}

variable "aws_gpu_node_types" {
  description = "GPU instance types for quantum acceleration"
  type = map(list(string))
  default = {
    "us-east-1"      = ["p4d.xlarge", "p4d.2xlarge", "p4d.8xlarge"]
    "eu-west-1"      = ["p3dn.24xlarge", "p4d.xlarge"]
    "ap-southeast-1" = ["p3.2xlarge", "p3.8xlarge"]
  }
}

# GCP Specific Variables
variable "gcp_project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "photonic-foundry-global"
}

variable "gcp_regions" {
  description = "GCP regions for deployment"
  type        = list(string)
  default     = ["us-central1", "europe-west1", "asia-southeast1"]
}

variable "gcp_machine_types" {
  description = "Machine types for GCP Kubernetes nodes"
  type = map(object({
    general_compute = list(string)
    quantum_compute = list(string)
    gpu_compute    = list(string)
  }))
  default = {
    "us-central1" = {
      general_compute = ["n1-standard-4", "n1-standard-8", "n1-highmem-8"]
      quantum_compute = ["c2-standard-16", "c2-standard-30", "c2-standard-60"]
      gpu_compute    = ["n1-standard-4", "n1-standard-8"]
    }
    "europe-west1" = {
      general_compute = ["n1-standard-4", "n1-standard-8", "n1-highmem-8"]
      quantum_compute = ["c2-standard-16", "c2-standard-30"]
      gpu_compute    = ["n1-standard-4", "n1-standard-8"]
    }
    "asia-southeast1" = {
      general_compute = ["n1-standard-2", "n1-standard-4", "n1-standard-8"]
      quantum_compute = ["c2-standard-8", "c2-standard-16"]
      gpu_compute    = ["n1-standard-4"]
    }
  }
}

# Azure Specific Variables
variable "azure_regions" {
  description = "Azure regions for deployment"
  type        = list(string)
  default     = ["East US", "West Europe", "Southeast Asia"]
}

variable "azure_vm_sizes" {
  description = "VM sizes for Azure Kubernetes nodes"
  type = map(object({
    general_compute = list(string)
    quantum_compute = list(string)
    gpu_compute    = list(string)
  }))
  default = {
    "East US" = {
      general_compute = ["Standard_D4s_v3", "Standard_D8s_v3", "Standard_D16s_v3"]
      quantum_compute = ["Standard_F16s_v2", "Standard_F32s_v2", "Standard_F64s_v2"]
      gpu_compute    = ["Standard_NC6s_v3", "Standard_NC12s_v3", "Standard_NC24s_v3"]
    }
    "West Europe" = {
      general_compute = ["Standard_D4s_v3", "Standard_D8s_v3", "Standard_D16s_v3"]
      quantum_compute = ["Standard_F16s_v2", "Standard_F32s_v2"]
      gpu_compute    = ["Standard_NC6s_v3", "Standard_NC12s_v3"]
    }
    "Southeast Asia" = {
      general_compute = ["Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3"]
      quantum_compute = ["Standard_F8s_v2", "Standard_F16s_v2"]
      gpu_compute    = ["Standard_NC6s_v3"]
    }
  }
}

# Alibaba Cloud Specific Variables
variable "alibaba_regions" {
  description = "Alibaba Cloud regions for deployment"
  type        = list(string)
  default     = ["ap-southeast-1", "ap-northeast-1"]
}

variable "alibaba_instance_types" {
  description = "Instance types for Alibaba Cloud Kubernetes nodes"
  type = map(object({
    general_compute = list(string)
    quantum_compute = list(string)
    gpu_compute    = list(string)
  }))
  default = {
    "ap-southeast-1" = {
      general_compute = ["ecs.g6.xlarge", "ecs.g6.2xlarge", "ecs.g6.4xlarge"]
      quantum_compute = ["ecs.c6.4xlarge", "ecs.c6.8xlarge", "ecs.c6.16xlarge"]
      gpu_compute    = ["ecs.gn6i-c4g1.xlarge", "ecs.gn6i-c8g1.2xlarge"]
    }
    "ap-northeast-1" = {
      general_compute = ["ecs.g6.large", "ecs.g6.xlarge", "ecs.g6.2xlarge"]
      quantum_compute = ["ecs.c6.2xlarge", "ecs.c6.4xlarge", "ecs.c6.8xlarge"]
      gpu_compute    = ["ecs.gn6i-c4g1.xlarge"]
    }
  }
}

# Networking Configuration
variable "enable_service_mesh" {
  description = "Enable service mesh across all clouds"
  type        = bool
  default     = true
}

variable "service_mesh_type" {
  description = "Service mesh implementation (istio, linkerd, consul)"
  type        = string
  default     = "istio"
}

variable "enable_cross_cloud_networking" {
  description = "Enable cross-cloud networking with VPN/peering"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_quantum_encryption" {
  description = "Enable quantum-resistant encryption"
  type        = bool
  default     = true
}

variable "compliance_frameworks" {
  description = "Compliance frameworks to enforce"
  type        = list(string)
  default     = ["gdpr", "ccpa", "pdpa"]
}

variable "enable_data_residency" {
  description = "Enforce data residency requirements"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_global_monitoring" {
  description = "Enable global monitoring and observability"
  type        = bool
  default     = true
}

variable "monitoring_retention_days" {
  description = "Metrics and logs retention period in days"
  type        = number
  default     = 30
}

variable "enable_cost_optimization" {
  description = "Enable automated cost optimization"
  type        = bool
  default     = true
}

# Backup and Disaster Recovery
variable "enable_cross_cloud_backup" {
  description = "Enable cross-cloud backup and replication"
  type        = bool
  default     = true
}

variable "backup_retention_years" {
  description = "Backup retention period in years for compliance"
  type        = number
  default     = 7
}

variable "disaster_recovery_rto" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
}

variable "disaster_recovery_rpo" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
}

# Application Configuration
variable "application_replicas" {
  description = "Number of application replicas per region"
  type = map(number)
  default = {
    "us-east-1"      = 5
    "eu-west-1"      = 4
    "ap-southeast-1" = 3
    "us-central1"    = 3
    "europe-west1"   = 3
    "East US"        = 3
    "West Europe"    = 3
  }
}

variable "enable_auto_scaling" {
  description = "Enable horizontal pod autoscaling"
  type        = bool
  default     = true
}

variable "max_replicas_per_region" {
  description = "Maximum replicas per region for autoscaling"
  type = map(number)
  default = {
    "us-east-1"      = 50
    "eu-west-1"      = 40
    "ap-southeast-1" = 25
    "us-central1"    = 30
    "europe-west1"   = 30
    "East US"        = 30
    "West Europe"    = 30
  }
}

# Development and Testing
variable "enable_staging_environment" {
  description = "Enable staging environment deployment"
  type        = bool
  default     = true
}

variable "enable_development_environment" {
  description = "Enable development environment deployment"
  type        = bool
  default     = false
}

# Feature Flags
variable "enable_quantum_simulation" {
  description = "Enable quantum circuit simulation capabilities"
  type        = bool
  default     = true
}

variable "enable_photonic_acceleration" {
  description = "Enable photonic neural network acceleration"
  type        = bool
  default     = true
}

variable "enable_edge_computing" {
  description = "Enable edge computing nodes"
  type        = bool
  default     = true
}

variable "enable_ai_optimization" {
  description = "Enable AI-powered resource optimization"
  type        = bool
  default     = true
}

# Resource Limits
variable "global_resource_quota" {
  description = "Global resource quotas across all clouds"
  type = object({
    total_cpu_cores    = number
    total_memory_gb    = number
    total_storage_tb   = number
    total_gpu_count    = number
  })
  default = {
    total_cpu_cores  = 1000
    total_memory_gb  = 4000
    total_storage_tb = 100
    total_gpu_count  = 50
  }
}

variable "cost_budget_monthly" {
  description = "Monthly cost budget across all clouds in USD"
  type        = number
  default     = 50000
}

# Validation
variable "validate_compliance" {
  description = "Validate compliance requirements during deployment"
  type        = bool
  default     = true
}

variable "validate_security" {
  description = "Validate security configurations during deployment"
  type        = bool
  default     = true
}

variable "validate_performance" {
  description = "Validate performance requirements during deployment"
  type        = bool
  default     = true
}
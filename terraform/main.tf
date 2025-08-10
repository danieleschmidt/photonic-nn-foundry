# Photonic Foundry Global Infrastructure - Main Configuration
# Multi-cloud deployment supporting AWS, GCP, Azure, and Alibaba Cloud

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.200"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  backend "s3" {
    # This will be configured per environment
    # bucket = "photonic-foundry-terraform-state"
    # key    = "global/terraform.tfstate"
    # region = "us-east-1"
    # encrypt = true
    # dynamodb_table = "photonic-foundry-terraform-locks"
  }
}

# Local values for shared configuration
locals {
  project_name = "photonic-foundry"
  environment  = var.environment
  
  # Global tags applied to all resources
  global_tags = {
    Project                = local.project_name
    Environment           = local.environment
    ManagedBy            = "Terraform"
    ComplianceFramework  = "multi-framework"
    QuantumCompute       = "enabled"
    DataSovereignty      = "enforced"
    CreatedDate          = formatdate("YYYY-MM-DD", timestamp())
  }
  
  # Regional configuration
  regions = {
    aws = {
      us-east-1 = {
        provider     = "aws"
        region_name  = "us-east-1"
        compliance   = "ccpa"
        jurisdiction = "usa"
        primary      = true
      }
      eu-west-1 = {
        provider     = "aws"
        region_name  = "eu-west-1"
        compliance   = "gdpr"
        jurisdiction = "eu"
        primary      = false
      }
      ap-southeast-1 = {
        provider     = "aws"
        region_name  = "ap-southeast-1"
        compliance   = "pdpa"
        jurisdiction = "singapore"
        primary      = false
      }
    }
    gcp = {
      us-central1 = {
        provider     = "gcp"
        region_name  = "us-central1"
        compliance   = "ccpa"
        jurisdiction = "usa"
        primary      = false
      }
      europe-west1 = {
        provider     = "gcp"
        region_name  = "europe-west1"
        compliance   = "gdpr"
        jurisdiction = "eu"
        primary      = false
      }
    }
    azure = {
      East_US = {
        provider     = "azure"
        region_name  = "East US"
        compliance   = "ccpa"
        jurisdiction = "usa"
        primary      = false
      }
      West_Europe = {
        provider     = "azure"
        region_name  = "West Europe"
        compliance   = "gdpr"
        jurisdiction = "eu"
        primary      = false
      }
    }
    alibaba = {
      ap-southeast-1 = {
        provider     = "alibaba"
        region_name  = "ap-southeast-1"
        compliance   = "pdpa"
        jurisdiction = "singapore"
        primary      = false
      }
    }
  }
}

# Random ID for unique resource naming
resource "random_id" "global" {
  byte_length = 4
}

# Data sources for current configurations
data "aws_caller_identity" "current" {
  count = var.enable_aws ? 1 : 0
}

data "google_client_config" "current" {
  count = var.enable_gcp ? 1 : 0
}

data "azurerm_client_config" "current" {
  count = var.enable_azure ? 1 : 0
}

# Global DNS and load balancing (using Cloudflare)
resource "cloudflare_zone" "main" {
  count = var.enable_cloudflare ? 1 : 0
  zone  = var.domain_name
  plan  = "pro"
  
  lifecycle {
    prevent_destroy = true
  }
}

# Global SSL certificate
resource "cloudflare_origin_ca_certificate" "main" {
  count      = var.enable_cloudflare ? 1 : 0
  csr        = tls_cert_request.main[0].cert_request_pem
  hostnames  = [var.domain_name, "*.${var.domain_name}"]
  request_type = "origin-rsa"
}

resource "tls_private_key" "main" {
  count     = var.enable_cloudflare ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_cert_request" "main" {
  count           = var.enable_cloudflare ? 1 : 0
  private_key_pem = tls_private_key.main[0].private_key_pem

  subject {
    common_name  = var.domain_name
    organization = "Photonic Foundry"
  }

  dns_names = [
    var.domain_name,
    "*.${var.domain_name}",
    "api.${var.domain_name}",
    "quantum.${var.domain_name}",
    "compliance.${var.domain_name}"
  ]
}

# Multi-cloud deployment modules
module "aws_infrastructure" {
  count = var.enable_aws ? 1 : 0
  
  source = "./modules/aws"
  
  project_name = local.project_name
  environment  = local.environment
  global_tags  = local.global_tags
  
  regions = local.regions.aws
  
  # Kubernetes configuration
  kubernetes_version = var.kubernetes_version
  node_instance_types = var.aws_node_instance_types
  
  # Compliance settings
  enable_compliance_mode = true
  compliance_frameworks  = ["ccpa", "gdpr"]
  
  # Quantum compute settings
  enable_quantum_nodes = var.enable_quantum_compute
  gpu_node_types      = var.aws_gpu_node_types
  
  # Networking
  vpc_cidr_blocks = var.aws_vpc_cidr_blocks
  
  # Monitoring
  enable_monitoring = true
  
  depends_on = [random_id.global]
}

module "gcp_infrastructure" {
  count = var.enable_gcp ? 1 : 0
  
  source = "./modules/gcp"
  
  project_name = local.project_name
  environment  = local.environment
  global_tags  = local.global_tags
  
  regions = local.regions.gcp
  
  # GCP specific settings
  project_id = var.gcp_project_id
  
  # Kubernetes configuration
  kubernetes_version = var.kubernetes_version
  
  # Compliance settings
  enable_compliance_mode = true
  compliance_frameworks  = ["gdpr", "ccpa"]
  
  depends_on = [random_id.global]
}

module "azure_infrastructure" {
  count = var.enable_azure ? 1 : 0
  
  source = "./modules/azure"
  
  project_name = local.project_name
  environment  = local.environment
  global_tags  = local.global_tags
  
  regions = local.regions.azure
  
  # Kubernetes configuration
  kubernetes_version = var.kubernetes_version
  
  # Compliance settings
  enable_compliance_mode = true
  compliance_frameworks  = ["gdpr", "ccpa"]
  
  depends_on = [random_id.global]
}

module "alibaba_infrastructure" {
  count = var.enable_alibaba ? 1 : 0
  
  source = "./modules/alibaba"
  
  project_name = local.project_name
  environment  = local.environment
  global_tags  = local.global_tags
  
  regions = local.regions.alibaba
  
  # Kubernetes configuration
  kubernetes_version = var.kubernetes_version
  
  # Compliance settings
  enable_compliance_mode = true
  compliance_frameworks  = ["pdpa"]
  
  depends_on = [random_id.global]
}

# Global monitoring and observability
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name = local.project_name
  environment  = local.environment
  
  # Multi-cloud endpoints
  aws_clusters    = var.enable_aws ? module.aws_infrastructure[0].cluster_endpoints : []
  gcp_clusters    = var.enable_gcp ? module.gcp_infrastructure[0].cluster_endpoints : []
  azure_clusters  = var.enable_azure ? module.azure_infrastructure[0].cluster_endpoints : []
  alibaba_clusters = var.enable_alibaba ? module.alibaba_infrastructure[0].cluster_endpoints : []
  
  # Global monitoring settings
  retention_days = 30
  
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure,
    module.alibaba_infrastructure
  ]
}

# Global secrets management
module "secrets" {
  source = "./modules/secrets"
  
  project_name = local.project_name
  environment  = local.environment
  
  # Quantum encryption keys
  enable_quantum_encryption = true
  
  # Multi-cloud secret replication
  replicate_to_aws     = var.enable_aws
  replicate_to_gcp     = var.enable_gcp
  replicate_to_azure   = var.enable_azure
  replicate_to_alibaba = var.enable_alibaba
  
  depends_on = [random_id.global]
}

# Global backup and disaster recovery
module "backup" {
  source = "./modules/backup"
  
  project_name = local.project_name
  environment  = local.environment
  
  # Cross-cloud backup configuration
  primary_cloud   = "aws"
  secondary_cloud = "gcp"
  
  # Compliance retention
  retention_years = 7
  
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure,
    module.alibaba_infrastructure
  ]
}
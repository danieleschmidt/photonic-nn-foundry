# AWS Infrastructure Module for Photonic Foundry
# Multi-region AWS deployment with quantum compute and compliance features

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

locals {
  common_tags = merge(var.global_tags, {
    CloudProvider = "AWS"
    Module       = "aws-infrastructure"
  })
}

# AWS Provider configurations for each region
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = merge(local.common_tags, {
      Region      = "us-east-1"
      Compliance  = "ccpa"
      Jurisdiction = "usa"
    })
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = merge(local.common_tags, {
      Region      = "eu-west-1"
      Compliance  = "gdpr"
      Jurisdiction = "eu"
    })
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  
  default_tags {
    tags = merge(local.common_tags, {
      Region      = "ap-southeast-1"
      Compliance  = "pdpa"
      Jurisdiction = "singapore"
    })
  }
}

# Data sources
data "aws_availability_zones" "us_east_1" {
  provider = aws.us_east_1
  state    = "available"
}

data "aws_availability_zones" "eu_west_1" {
  provider = aws.eu_west_1
  state    = "available"
}

data "aws_availability_zones" "ap_southeast_1" {
  provider = aws.ap_southeast_1
  state    = "available"
}

# VPC for US East 1 (Primary Region)
module "vpc_us_east_1" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  providers = {
    aws = aws.us_east_1
  }
  
  name = "${var.project_name}-vpc-us-east-1"
  cidr = var.vpc_cidr_blocks["us-east-1"]
  
  azs             = slice(data.aws_availability_zones.us_east_1.names, 0, 3)
  private_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 1),
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 2),
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 3)
  ]
  public_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 101),
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 102),
    cidrsubnet(var.vpc_cidr_blocks["us-east-1"], 8, 103)
  ]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Compliance and security
  enable_flow_log                 = true
  flow_log_destination_type       = "cloud-watch-logs"
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${var.project_name}-us-east-1" = "shared"
    Region = "us-east-1"
    Primary = "true"
  })
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-us-east-1" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
    Tier = "private"
  }
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-us-east-1" = "owned"
    "kubernetes.io/role/elb" = "1"
    Tier = "public"
  }
}

# VPC for EU West 1 (GDPR Compliance)
module "vpc_eu_west_1" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  name = "${var.project_name}-vpc-eu-west-1"
  cidr = var.vpc_cidr_blocks["eu-west-1"]
  
  azs             = slice(data.aws_availability_zones.eu_west_1.names, 0, 3)
  private_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 1),
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 2),
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 3)
  ]
  public_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 101),
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 102),
    cidrsubnet(var.vpc_cidr_blocks["eu-west-1"], 8, 103)
  ]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # GDPR compliance features
  enable_flow_log                 = true
  flow_log_destination_type       = "cloud-watch-logs"
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${var.project_name}-eu-west-1" = "shared"
    Region = "eu-west-1"
    GDPRCompliant = "true"
    DataSovereignty = "eu"
  })
}

# VPC for AP Southeast 1 (PDPA Compliance)
module "vpc_ap_southeast_1" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  providers = {
    aws = aws.ap_southeast_1
  }
  
  name = "${var.project_name}-vpc-ap-southeast-1"
  cidr = var.vpc_cidr_blocks["ap-southeast-1"]
  
  azs             = slice(data.aws_availability_zones.ap_southeast_1.names, 0, 3)
  private_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 1),
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 2),
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 3)
  ]
  public_subnets = [
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 101),
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 102),
    cidrsubnet(var.vpc_cidr_blocks["ap-southeast-1"], 8, 103)
  ]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false  # Reduced infrastructure for cost optimization
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # PDPA compliance features
  enable_flow_log                 = true
  flow_log_destination_type       = "cloud-watch-logs"
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${var.project_name}-ap-southeast-1" = "shared"
    Region = "ap-southeast-1"
    PDPACompliant = "true"
    DataSovereignty = "singapore"
  })
}

# EKS Clusters
module "eks_us_east_1" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  providers = {
    aws = aws.us_east_1
  }
  
  cluster_name    = "${var.project_name}-us-east-1"
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc_us_east_1.vpc_id
  subnet_ids                     = module.vpc_us_east_1.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true
  
  # Enable logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks_us_east_1.arn
    resources        = ["secrets"]
  }
  
  # Node groups
  eks_managed_node_groups = {
    general = {
      name = "general-compute"
      
      instance_types = var.node_instance_types["us-east-1"].general_compute
      
      min_size     = 3
      max_size     = 20
      desired_size = 5
      
      labels = {
        WorkloadType = "general"
        Region      = "us-east-1"
      }
      
      taints = []
    }
    
    quantum = {
      name = "quantum-compute"
      
      instance_types = var.node_instance_types["us-east-1"].quantum_compute
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      labels = {
        WorkloadType = "quantum"
        Region      = "us-east-1"
        QuantumCompute = "true"
      }
      
      taints = [
        {
          key    = "quantum-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    gpu = {
      name = "gpu-compute"
      
      instance_types = var.gpu_node_types["us-east-1"]
      
      min_size     = 0
      max_size     = 5
      desired_size = 1
      
      labels = {
        WorkloadType = "gpu"
        Region      = "us-east-1"
        GPUAcceleration = "true"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  tags = merge(local.common_tags, {
    Region = "us-east-1"
    Primary = "true"
  })
}

# Similar EKS clusters for other regions (abbreviated for space)
module "eks_eu_west_1" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  cluster_name    = "${var.project_name}-eu-west-1"
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc_eu_west_1.vpc_id
  subnet_ids                     = module.vpc_eu_west_1.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true
  
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks_eu_west_1.arn
    resources        = ["secrets"]
  }
  
  # GDPR-compliant node configuration
  eks_managed_node_groups = {
    general = {
      name = "general-compute-gdpr"
      instance_types = var.node_instance_types["eu-west-1"].general_compute
      min_size = 2
      max_size = 15
      desired_size = 4
      
      labels = {
        WorkloadType = "general"
        Region = "eu-west-1"
        GDPRCompliant = "true"
      }
    }
    
    quantum = {
      name = "quantum-compute-gdpr"
      instance_types = var.node_instance_types["eu-west-1"].quantum_compute
      min_size = 1
      max_size = 8
      desired_size = 2
      
      labels = {
        WorkloadType = "quantum"
        Region = "eu-west-1"
        QuantumCompute = "true"
        GDPRCompliant = "true"
      }
    }
  }
  
  tags = merge(local.common_tags, {
    Region = "eu-west-1"
    GDPRCompliant = "true"
  })
}

# KMS Keys for encryption
resource "aws_kms_key" "eks_us_east_1" {
  provider = aws.us_east_1
  
  description         = "EKS Encryption Key - US East 1"
  enable_key_rotation = true
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-key-us-east-1"
    Region = "us-east-1"
  })
}

resource "aws_kms_key" "eks_eu_west_1" {
  provider = aws.eu_west_1
  
  description         = "EKS Encryption Key - EU West 1"
  enable_key_rotation = true
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-key-eu-west-1"
    Region = "eu-west-1"
  })
}

resource "aws_kms_key" "eks_ap_southeast_1" {
  provider = aws.ap_southeast_1
  
  description         = "EKS Encryption Key - AP Southeast 1"
  enable_key_rotation = true
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-eks-key-ap-southeast-1"
    Region = "ap-southeast-1"
  })
}

# CloudWatch Log Groups for compliance
resource "aws_cloudwatch_log_group" "eks_us_east_1" {
  provider = aws.us_east_1
  
  name              = "/aws/eks/${var.project_name}-us-east-1/cluster"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.eks_us_east_1.arn
  
  tags = merge(local.common_tags, {
    Region = "us-east-1"
    Purpose = "eks-logging"
  })
}

# Application Load Balancers for each region
resource "aws_lb" "main_us_east_1" {
  provider = aws.us_east_1
  
  name               = "${var.project_name}-alb-us-east-1"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_us_east_1.id]
  subnets            = module.vpc_us_east_1.public_subnets
  
  enable_deletion_protection = true
  enable_http2              = true
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs_us_east_1.bucket
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = merge(local.common_tags, {
    Region = "us-east-1"
    Purpose = "application-load-balancer"
  })
}

# Security Groups
resource "aws_security_group" "alb_us_east_1" {
  provider = aws.us_east_1
  
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = module.vpc_us_east_1.vpc_id
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name   = "${var.project_name}-alb-sg-us-east-1"
    Region = "us-east-1"
  })
}

# S3 Buckets for ALB logs
resource "aws_s3_bucket" "alb_logs_us_east_1" {
  provider = aws.us_east_1
  
  bucket = "${var.project_name}-alb-logs-us-east-1-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Region = "us-east-1"
    Purpose = "alb-logs"
  })
}

resource "aws_s3_bucket_versioning" "alb_logs_us_east_1" {
  provider = aws.us_east_1
  
  bucket = aws_s3_bucket.alb_logs_us_east_1.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "alb_logs_us_east_1" {
  provider = aws.us_east_1
  
  bucket = aws_s3_bucket.alb_logs_us_east_1.id
  
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.eks_us_east_1.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}
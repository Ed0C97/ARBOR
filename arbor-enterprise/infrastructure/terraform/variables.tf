# =============================================================================
# A.R.B.O.R. Enterprise - Terraform Variables
# =============================================================================

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_name" {
  description = "Project name used as prefix for resources"
  type        = string
  default     = "arbor-enterprise"
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (staging, production)"
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

# -----------------------------------------------------------------------------
# VPC Networking
# -----------------------------------------------------------------------------

variable "gke_subnet_cidr" {
  description = "Primary CIDR range for GKE subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary CIDR range for GKE pods"
  type        = string
  default     = "10.16.0.0/14"
}

variable "services_cidr" {
  description = "Secondary CIDR range for GKE services"
  type        = string
  default     = "10.20.0.0/20"
}

variable "master_cidr" {
  description = "CIDR range for GKE master nodes (private cluster)"
  type        = string
  default     = "172.16.0.0/28"
}

# -----------------------------------------------------------------------------
# GKE Cluster
# -----------------------------------------------------------------------------

variable "gke_release_channel" {
  description = "GKE release channel (RAPID, REGULAR, STABLE)"
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.gke_release_channel)
    error_message = "Release channel must be RAPID, REGULAR, or STABLE."
  }
}

# API node pool
variable "api_pool_machine_type" {
  description = "Machine type for API node pool"
  type        = string
  default     = "e2-standard-4"
}

variable "api_pool_min_count" {
  description = "Minimum number of nodes in API pool"
  type        = number
  default     = 2
}

variable "api_pool_max_count" {
  description = "Maximum number of nodes in API pool"
  type        = number
  default     = 10
}

variable "api_pool_disk_size_gb" {
  description = "Disk size in GB for API pool nodes"
  type        = number
  default     = 50
}

# Worker node pool
variable "worker_pool_machine_type" {
  description = "Machine type for Worker node pool"
  type        = string
  default     = "e2-standard-8"
}

variable "worker_pool_min_count" {
  description = "Minimum number of nodes in Worker pool"
  type        = number
  default     = 1
}

variable "worker_pool_max_count" {
  description = "Maximum number of nodes in Worker pool"
  type        = number
  default     = 8
}

variable "worker_pool_disk_size_gb" {
  description = "Disk size in GB for Worker pool nodes"
  type        = number
  default     = 100
}

# -----------------------------------------------------------------------------
# Cloud SQL PostgreSQL
# -----------------------------------------------------------------------------

variable "cloudsql_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-custom-4-16384"
}

variable "cloudsql_disk_size_gb" {
  description = "Cloud SQL disk size in GB"
  type        = number
  default     = 50
}

variable "cloudsql_max_connections" {
  description = "Maximum database connections"
  type        = string
  default     = "200"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "arbor"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# -----------------------------------------------------------------------------
# Redis Memorystore
# -----------------------------------------------------------------------------

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 2
}

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------

variable "app_version" {
  description = "Application version / Docker image tag"
  type        = string
  default     = "latest"
}

variable "api_replicas" {
  description = "Number of API pod replicas"
  type        = number
  default     = 3
}

variable "worker_replicas" {
  description = "Number of Worker pod replicas"
  type        = number
  default     = 2
}

variable "api_cpu_request" {
  description = "CPU request for API pods"
  type        = string
  default     = "500m"
}

variable "api_cpu_limit" {
  description = "CPU limit for API pods"
  type        = string
  default     = "2000m"
}

variable "api_memory_request" {
  description = "Memory request for API pods"
  type        = string
  default     = "512Mi"
}

variable "api_memory_limit" {
  description = "Memory limit for API pods"
  type        = string
  default     = "2Gi"
}

variable "worker_cpu_request" {
  description = "CPU request for Worker pods"
  type        = string
  default     = "1000m"
}

variable "worker_cpu_limit" {
  description = "CPU limit for Worker pods"
  type        = string
  default     = "4000m"
}

variable "worker_memory_request" {
  description = "Memory request for Worker pods"
  type        = string
  default     = "1Gi"
}

variable "worker_memory_limit" {
  description = "Memory limit for Worker pods"
  type        = string
  default     = "4Gi"
}

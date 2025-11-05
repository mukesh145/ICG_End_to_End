variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-3"
}

variable "app_name" {
  description = "Logical app name"
  type        = string
  default     = "icg-api"
}

variable "image_uri" {
  description = "ECR image URI, e.g. 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/mnist-api:latest"
  type        = string
}

variable "container_port" {
  description = "Container port your API listens on"
  type        = number
  default     = 8080
}

variable "health_check_path" {
  description = "HTTP health check path for ALB"
  type        = string
  default     = "/health"
}

variable "task_cpu" {
  description = "Fargate CPU (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate Memory in MiB (512–30720 depending on CPU)"
  type        = number
  default     = 2048
}

variable "cpu_architecture" {
  description = "CPU arch: X86_64 or ARM64"
  type        = string
  default     = "X86_64"
}

variable "ephemeral_storage_gib" {
  description = "Fargate ephemeral storage in GiB (20–200)"
  type        = number
  default     = 100
  validation {
    condition     = var.ephemeral_storage_gib >= 20 && var.ephemeral_storage_gib <= 200
    error_message = "ephemeral_storage_gib must be between 20 and 200 GiB."
  }
}

variable "desired_count" {
  description = "Number of tasks"
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch logs retention"
  type        = number
  default     = 14
}

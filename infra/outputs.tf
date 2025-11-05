output "alb_dns_name" {
  value       = aws_lb.this.dns_name
  description = "Public DNS of the ALB"
}

output "service_name" {
  value       = aws_ecs_service.this.name
  description = "ECS service name"
}

output "cluster_name" {
  value       = aws_ecs_cluster.this.name
  description = "ECS cluster name"
}

import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num) {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num?.toString() ?? '0';
}

export function formatLatency(ms) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms}ms`;
}

export function formatPercentage(value, decimals = 1) {
  return `${Number(value).toFixed(decimals)}%`;
}

export function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function truncate(str, maxLength = 100) {
  if (!str || str.length <= maxLength) return str;
  return str.slice(0, maxLength) + '...';
}

export function getEntityTypeColor(type) {
  switch (type) {
    case 'brand': return 'text-chart-1';
    case 'venue': return 'text-chart-2';
    default: return 'text-muted-foreground';
  }
}

export function getStatusColor(status) {
  switch (status) {
    case 'healthy': return 'text-success';
    case 'degraded': return 'text-warning';
    case 'unhealthy': return 'text-destructive';
    case 'approved': return 'text-success';
    case 'rejected': return 'text-destructive';
    case 'needs_review': return 'text-warning';
    default: return 'text-muted-foreground';
  }
}

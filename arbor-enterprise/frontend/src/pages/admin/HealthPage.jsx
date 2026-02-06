import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  HeartPulse,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Server,
  Loader2,
  Timer,
  Info,
} from 'lucide-react';

import { cn, formatLatency } from '@/lib/utils';
import { useAdminStore } from '@/stores/adminStore';
import { apiGet } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';

const REFRESH_INTERVAL = 30_000; // 30 seconds

function getStatusIcon(status) {
  switch (status) {
    case 'healthy':
      return <CheckCircle2 className="size-4 text-emerald-500" />;
    case 'degraded':
      return <AlertTriangle className="size-4 text-amber-500" />;
    case 'unhealthy':
      return <XCircle className="size-4 text-destructive" />;
    default:
      return <Info className="size-4 text-muted-foreground" />;
  }
}

function getStatusBadgeVariant(status) {
  switch (status) {
    case 'healthy':
      return 'success';
    case 'degraded':
      return 'warning';
    case 'unhealthy':
      return 'destructive';
    default:
      return 'secondary';
  }
}

function ServiceCheckCard({ check }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card
        className={cn(
          'cursor-pointer transition-colors',
          check.status === 'unhealthy' && 'border-destructive/30',
          check.status === 'degraded' && 'border-amber-500/30',
        )}
        onClick={() => setExpanded(!expanded)}
      >
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getStatusIcon(check.status)}
              <div>
                <p className="text-sm font-medium">{check.name}</p>
                {check.latency_ms !== undefined && (
                  <p className="text-xs text-muted-foreground">
                    {formatLatency(check.latency_ms)}
                  </p>
                )}
              </div>
            </div>
            <Badge variant={getStatusBadgeVariant(check.status)} className="capitalize">
              {check.status}
            </Badge>
          </div>

          {expanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-3 space-y-2 border-t pt-3"
            >
              {check.error && (
                <div className="flex items-start gap-2">
                  <AlertCircle className="mt-0.5 size-3 shrink-0 text-destructive" />
                  <p className="text-xs text-destructive">{check.error}</p>
                </div>
              )}
              {check.details && (
                <div className="space-y-1">
                  {typeof check.details === 'object' ? (
                    Object.entries(check.details).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">{key}</span>
                        <span className="font-mono">{String(value)}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-muted-foreground">{String(check.details)}</p>
                  )}
                </div>
              )}
              {!check.error && !check.details && (
                <p className="text-xs text-muted-foreground">No additional details available</p>
              )}
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default function AdminHealthPage() {
  const { health, healthLoading, healthError, fetchHealth } = useAdminStore();

  const [liveness, setLiveness] = useState(null);
  const [livenessLoading, setLivenessLoading] = useState(false);
  const [refreshTimer, setRefreshTimer] = useState(REFRESH_INTERVAL / 1000);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const timerRef = useRef(null);
  const intervalRef = useRef(null);

  const loadAll = useCallback(async () => {
    fetchHealth();
    setLivenessLoading(true);
    try {
      const data = await apiGet('/api/v1/admin/health/liveness');
      setLiveness(data);
    } catch {
      setLiveness(null);
    } finally {
      setLivenessLoading(false);
    }
  }, [fetchHealth]);

  // Initial load
  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // Auto-refresh timer
  useEffect(() => {
    if (!autoRefresh) {
      clearInterval(timerRef.current);
      clearTimeout(intervalRef.current);
      return;
    }

    setRefreshTimer(REFRESH_INTERVAL / 1000);

    timerRef.current = setInterval(() => {
      setRefreshTimer((prev) => {
        if (prev <= 1) {
          loadAll();
          return REFRESH_INTERVAL / 1000;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      clearInterval(timerRef.current);
    };
  }, [autoRefresh, loadAll]);

  function handleManualRefresh() {
    loadAll();
    setRefreshTimer(REFRESH_INTERVAL / 1000);
  }

  const checks = health?.checks || [];
  const healthyCount = checks.filter((c) => c.status === 'healthy').length;
  const degradedCount = checks.filter((c) => c.status === 'degraded').length;
  const unhealthyCount = checks.filter((c) => c.status === 'unhealthy').length;

  const timerProgress = (refreshTimer / (REFRESH_INTERVAL / 1000)) * 100;

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <HeartPulse className="size-5" />
            System Health
          </h2>
          <p className="text-sm text-muted-foreground">
            Real-time monitoring of all service components
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={autoRefresh ? 'secondary' : 'outline'}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <Timer className="size-4" />
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleManualRefresh}
            disabled={healthLoading}
          >
            <RefreshCw className={cn('size-4', healthLoading && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Auto-refresh Timer */}
      {autoRefresh && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Clock className="size-3" />
              Next refresh in {refreshTimer}s
            </span>
          </div>
          <Progress value={timerProgress} className="h-1" />
        </div>
      )}

      <Separator />

      {/* Error State */}
      {healthError && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-center gap-3 p-4">
            <AlertCircle className="size-5 text-destructive" />
            <div>
              <p className="text-sm font-medium">Failed to fetch health status</p>
              <p className="text-xs text-muted-foreground">{healthError}</p>
            </div>
            <Button variant="outline" size="sm" className="ml-auto" onClick={handleManualRefresh}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Overview Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Overall Status */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Overall Status</CardTitle>
            <Server className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {healthLoading && !health ? (
              <Skeleton className="h-8 w-24" />
            ) : health ? (
              <Badge
                variant={getStatusBadgeVariant(health.status)}
                className="text-base capitalize px-3 py-1"
              >
                {health.status}
              </Badge>
            ) : (
              <span className="text-sm text-muted-foreground">Unknown</span>
            )}
          </CardContent>
        </Card>

        {/* Liveness */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Liveness</CardTitle>
            <HeartPulse className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {livenessLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : liveness ? (
              <Badge
                variant={liveness.status === 'alive' || liveness.status === 'ok' ? 'success' : 'destructive'}
                className="text-base capitalize px-3 py-1"
              >
                {liveness.status}
              </Badge>
            ) : (
              <span className="text-sm text-muted-foreground">Unreachable</span>
            )}
          </CardContent>
        </Card>

        {/* Version */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Version</CardTitle>
            <Info className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {healthLoading && !health ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className="text-2xl font-bold font-mono">{health?.version || 'N/A'}</p>
            )}
          </CardContent>
        </Card>

        {/* Environment */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Environment</CardTitle>
            <Server className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {healthLoading && !health ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className="text-2xl font-bold capitalize">{health?.environment || 'N/A'}</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Total Latency & Check Summary */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10">
              <Clock className="size-5 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold font-mono">
                {health?.total_latency_ms !== undefined
                  ? formatLatency(health.total_latency_ms)
                  : '--'}
              </p>
              <p className="text-xs text-muted-foreground">Total Latency</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-emerald-500/10">
              <CheckCircle2 className="size-5 text-emerald-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{healthyCount}</p>
              <p className="text-xs text-muted-foreground">Healthy Services</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-destructive/10">
              <AlertCircle className="size-5 text-destructive" />
            </div>
            <div>
              <p className="text-2xl font-bold">{degradedCount + unhealthyCount}</p>
              <p className="text-xs text-muted-foreground">Issues Detected</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Service Checks */}
      <div>
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
          <Server className="size-4" />
          Service Checks
          {checks.length > 0 && (
            <span className="text-xs font-normal text-muted-foreground">
              ({checks.length} services)
            </span>
          )}
        </h3>

        {healthLoading && checks.length === 0 ? (
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : checks.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Server className="mb-3 size-8 text-muted-foreground/50" />
              <p className="text-sm font-medium">No service checks available</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Health data will appear once the backend is reachable
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {/* Sort: unhealthy first, then degraded, then healthy */}
            {[...checks]
              .sort((a, b) => {
                const order = { unhealthy: 0, degraded: 1, healthy: 2 };
                return (order[a.status] ?? 3) - (order[b.status] ?? 3);
              })
              .map((check, i) => (
                <ServiceCheckCard key={check.name || i} check={check} />
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

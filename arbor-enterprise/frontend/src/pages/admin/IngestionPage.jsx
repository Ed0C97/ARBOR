import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Sparkles,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Clock,
  RefreshCw,
  Play,
  XCircle,
  Database,
  Zap,
} from 'lucide-react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { useAdminStore } from '@/stores/adminStore';
import { apiGet } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';

function EnrichmentStatusCard({ status }) {
  if (!status) return null;

  const statusColor = {
    running: 'text-primary',
    completed: 'text-emerald-500',
    failed: 'text-destructive',
    idle: 'text-muted-foreground',
    queued: 'text-amber-500',
  };

  const statusIcon = {
    running: <Loader2 className="size-4 animate-spin" />,
    completed: <CheckCircle2 className="size-4" />,
    failed: <XCircle className="size-4" />,
    idle: <Clock className="size-4" />,
    queued: <Clock className="size-4" />,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className={cn(statusColor[status.status] || 'text-muted-foreground')}>
                {statusIcon[status.status] || <Clock className="size-4" />}
              </span>
              <span className="text-sm font-medium">{status.name || status.task_id || 'Enrichment Task'}</span>
            </div>
            <Badge
              variant={
                status.status === 'completed' ? 'success' :
                status.status === 'failed' ? 'destructive' :
                status.status === 'running' ? 'default' :
                'secondary'
              }
              className="capitalize"
            >
              {status.status}
            </Badge>
          </div>
          {status.progress !== undefined && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{status.processed || 0} / {status.total || 0} entities</span>
                <span>{Math.round(status.progress || 0)}%</span>
              </div>
              <Progress value={status.progress || 0} />
            </div>
          )}
          {status.error && (
            <p className="mt-2 text-xs text-destructive">{status.error}</p>
          )}
          {status.started_at && (
            <p className="mt-2 text-xs text-muted-foreground">
              Started: {new Date(status.started_at).toLocaleString()}
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default function AdminIngestionPage() {
  const { enrichEntity, fetchEnrichmentStatus, enrichmentStatus, enrichmentLoading } = useAdminStore();

  const [entityId, setEntityId] = useState('');
  const [batchIds, setBatchIds] = useState('');
  const [singleLoading, setSingleLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [statusList, setStatusList] = useState([]);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState(null);

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    setStatusError(null);
    try {
      const data = await fetchEnrichmentStatus();
      // The status endpoint may return a single object or an array
      if (data) {
        setStatusList(Array.isArray(data.tasks) ? data.tasks : Array.isArray(data) ? data : [data]);
      }
    } catch (err) {
      setStatusError(err.message);
    } finally {
      setStatusLoading(false);
    }
  }, [fetchEnrichmentStatus]);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  async function handleSingleEnrich(e) {
    e.preventDefault();
    const id = entityId.trim();
    if (!id) {
      toast.error('Please enter an entity ID');
      return;
    }
    setSingleLoading(true);
    try {
      await enrichEntity(id);
      toast.success(`Enrichment triggered for entity ${id}`);
      setEntityId('');
      loadStatus();
    } catch (err) {
      toast.error(`Enrichment failed: ${err.message}`);
    } finally {
      setSingleLoading(false);
    }
  }

  async function handleBatchEnrich(e) {
    e.preventDefault();
    const ids = batchIds
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    if (ids.length === 0) {
      toast.error('Please enter at least one entity ID');
      return;
    }

    setBatchLoading(true);
    let successCount = 0;
    let failCount = 0;

    for (const id of ids) {
      try {
        await enrichEntity(id);
        successCount++;
      } catch {
        failCount++;
      }
    }

    setBatchLoading(false);
    if (successCount > 0) {
      toast.success(`Enrichment triggered for ${successCount} entities`);
    }
    if (failCount > 0) {
      toast.error(`${failCount} enrichment(s) failed`);
    }
    setBatchIds('');
    loadStatus();
  }

  const runningTasks = statusList.filter((s) => s.status === 'running' || s.status === 'queued');
  const completedTasks = statusList.filter((s) => s.status === 'completed');
  const failedTasks = statusList.filter((s) => s.status === 'failed');

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Upload className="size-5" />
            Data Ingestion
          </h2>
          <p className="text-sm text-muted-foreground">
            Trigger and monitor entity enrichment operations
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={loadStatus}
          disabled={statusLoading}
        >
          <RefreshCw className={cn('size-4', statusLoading && 'animate-spin')} />
          Refresh Status
        </Button>
      </div>

      <Separator />

      {/* Enrichment Forms */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Single Entity Enrichment */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-primary" />
              <CardTitle className="text-sm font-medium">Single Entity Enrichment</CardTitle>
            </div>
            <CardDescription>Trigger enrichment for a specific entity by ID</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSingleEnrich} className="space-y-4">
              <Input
                placeholder="Enter entity ID..."
                value={entityId}
                onChange={(e) => setEntityId(e.target.value)}
                disabled={singleLoading}
              />
              <Button type="submit" size="sm" disabled={singleLoading || !entityId.trim()}>
                {singleLoading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Play className="size-4" />
                )}
                Enrich Entity
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Batch Enrichment */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Zap className="size-4 text-primary" />
              <CardTitle className="text-sm font-medium">Batch Enrichment</CardTitle>
            </div>
            <CardDescription>Enrich multiple entities at once (comma or newline separated)</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleBatchEnrich} className="space-y-4">
              <textarea
                placeholder="Enter entity IDs (one per line or comma-separated)..."
                value={batchIds}
                onChange={(e) => setBatchIds(e.target.value)}
                disabled={batchLoading}
                rows={4}
                className="w-full resize-none rounded-md border bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <Button type="submit" size="sm" disabled={batchLoading || !batchIds.trim()}>
                {batchLoading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Zap className="size-4" />
                )}
                Enrich Batch
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>

      {/* Status Summary */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10">
              <Loader2 className="size-5 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold">{runningTasks.length}</p>
              <p className="text-xs text-muted-foreground">Running / Queued</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-emerald-500/10">
              <CheckCircle2 className="size-5 text-emerald-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">{completedTasks.length}</p>
              <p className="text-xs text-muted-foreground">Completed</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex size-10 items-center justify-center rounded-lg bg-destructive/10">
              <XCircle className="size-5 text-destructive" />
            </div>
            <div>
              <p className="text-2xl font-bold">{failedTasks.length}</p>
              <p className="text-xs text-muted-foreground">Failed</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Enrichment Status List */}
      <div>
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
          <Database className="size-4" />
          Enrichment Tasks
        </h3>

        {statusError && (
          <Card className="mb-4 border-destructive/50 bg-destructive/5">
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="size-5 text-destructive" />
              <div>
                <p className="text-sm font-medium">Failed to load status</p>
                <p className="text-xs text-muted-foreground">{statusError}</p>
              </div>
              <Button variant="outline" size="sm" className="ml-auto" onClick={loadStatus}>
                Retry
              </Button>
            </CardContent>
          </Card>
        )}

        {statusLoading && statusList.length === 0 ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : statusList.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Database className="mb-3 size-8 text-muted-foreground/50" />
              <p className="text-sm font-medium">No enrichment tasks</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Trigger an enrichment above to get started
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {statusList.map((task, i) => (
              <EnrichmentStatusCard key={task.task_id || task.id || i} status={task} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

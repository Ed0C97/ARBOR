import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import {
  Activity,
  RefreshCw,
  Calendar,
  Database,
  AlertTriangle,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  FileWarning,
  CheckCircle2,
  Loader2,
  ArrowRight,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { apiGet } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getDriftPercentage(drifted, total) {
  if (!total || total === 0) return 0;
  return (drifted / total) * 100;
}

function getDriftSeverityColor(percentage) {
  if (percentage <= 5) return 'text-emerald-500';
  if (percentage <= 15) return 'text-amber-500';
  if (percentage <= 30) return 'text-orange-500';
  return 'text-red-500';
}

function getDriftSeverityLabel(percentage) {
  if (percentage <= 5) return 'Low';
  if (percentage <= 15) return 'Moderate';
  if (percentage <= 30) return 'High';
  return 'Critical';
}

function getDriftSeverityBadgeVariant(percentage) {
  if (percentage <= 5) return 'success';
  if (percentage <= 15) return 'warning';
  return 'destructive';
}

function getDriftProgressColor(percentage) {
  if (percentage <= 5) return 'bg-emerald-500';
  if (percentage <= 15) return 'bg-amber-500';
  if (percentage <= 30) return 'bg-orange-500';
  return 'bg-red-500';
}

function formatReportDate(dateString) {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ---------------------------------------------------------------------------
// Stat Card
// ---------------------------------------------------------------------------

function StatCard({ icon: Icon, label, value, description, className }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg',
              className,
            )}
          >
            <Icon className="size-4" />
          </div>
          <div className="min-w-0">
            <p className="text-2xl font-bold tabular-nums">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Drift Detail Card
// ---------------------------------------------------------------------------

function DriftDetailCard({ detail }) {
  const [expanded, setExpanded] = useState(false);

  const fields = detail.fields || detail.changed_fields || [];
  const fieldCount = fields.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="transition-colors hover:bg-accent/20">
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <AlertTriangle className="size-3.5 shrink-0 text-amber-500" />
                <h4 className="truncate text-sm font-semibold">
                  {detail.entity_name || detail.entity_id}
                </h4>
                <Badge variant="outline" className="shrink-0 text-[10px] px-1.5 py-0 tabular-nums">
                  {fieldCount} {fieldCount === 1 ? 'field' : 'fields'}
                </Badge>
              </div>

              {detail.entity_id && detail.entity_name && (
                <p className="mt-0.5 text-xs text-muted-foreground">
                  ID: {detail.entity_id}
                </p>
              )}
            </div>

            {fieldCount > 0 && (
              <Button
                variant="ghost"
                size="icon"
                className="size-7 shrink-0"
                onClick={() => setExpanded((prev) => !prev)}
              >
                {expanded ? (
                  <ChevronUp className="size-3.5" />
                ) : (
                  <ChevronDown className="size-3.5" />
                )}
              </Button>
            )}
          </div>

          {/* Expanded field details */}
          {expanded && fields.length > 0 && (
            <div className="mt-3 space-y-2">
              <Separator />
              {fields.map((field, idx) => {
                const fieldName =
                  typeof field === 'string' ? field : field.field || field.name;
                const oldValue =
                  typeof field === 'object' ? field.old_value || field.expected : null;
                const newValue =
                  typeof field === 'object' ? field.new_value || field.actual : null;

                return (
                  <div
                    key={idx}
                    className="rounded-md bg-muted/50 px-3 py-2 text-xs"
                  >
                    <span className="font-medium text-foreground capitalize">
                      {fieldName}
                    </span>

                    {(oldValue !== null || newValue !== null) && (
                      <div className="mt-1 flex items-center gap-2 text-muted-foreground">
                        <span className="line-through">{String(oldValue ?? '--')}</span>
                        <ArrowRight className="size-3 shrink-0" />
                        <span className="font-medium text-foreground">
                          {String(newValue ?? '--')}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton Loader
// ---------------------------------------------------------------------------

function ReportSkeleton() {
  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Skeleton className="h-9 w-9 rounded-lg" />
                <div className="space-y-1.5">
                  <Skeleton className="h-6 w-16" />
                  <Skeleton className="h-3 w-24" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Progress */}
      <Card>
        <CardContent className="p-5 space-y-3">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-full rounded-full" />
          <Skeleton className="h-3 w-40" />
        </CardContent>
      </Card>

      {/* Details */}
      <div className="space-y-3">
        <Skeleton className="h-4 w-28" />
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <Skeleton className="h-4 w-48" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DriftReportPage() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // -----------------------------------
  // Fetch report
  // -----------------------------------
  const fetchReport = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);

      const data = await apiGet('/api/v1/curator/drift-report');
      setReport(data);
    } catch (err) {
      if (err.name !== 'AbortError') {
        toast.error('Failed to load drift report');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  // Derived values
  const driftPercentage = report
    ? getDriftPercentage(report.drifted_entities, report.total_entities_checked)
    : 0;
  const driftDetails = report?.drift_details || [];

  // -----------------------------------
  // Render
  // -----------------------------------
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-violet-500/10 text-violet-500">
            <Activity className="size-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Drift Report</h2>
            <p className="text-sm text-muted-foreground">
              Monitor data consistency across entities
            </p>
          </div>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => fetchReport(true)}
          disabled={refreshing}
        >
          <RefreshCw
            className={cn('mr-1.5 size-3.5', refreshing && 'animate-spin')}
          />
          Refresh
        </Button>
      </div>

      <Separator />

      {/* Content */}
      {loading ? (
        <ReportSkeleton />
      ) : !report ? (
        /* Error / No data state */
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-muted text-muted-foreground">
            <FileWarning className="size-8" />
          </div>
          <h3 className="text-sm font-medium">No drift report available</h3>
          <p className="mt-1 max-w-xs text-xs text-muted-foreground">
            Drift reports are generated periodically. Try refreshing later.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Report date */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Calendar className="size-3.5" />
            <span>Report generated: {formatReportDate(report.report_date)}</span>
          </div>

          {/* Stat cards */}
          <div className="grid gap-4 sm:grid-cols-3">
            <StatCard
              icon={Database}
              label="Total Entities Checked"
              value={report.total_entities_checked ?? 0}
              className="bg-blue-500/10 text-blue-500"
            />
            <StatCard
              icon={AlertTriangle}
              label="Drifted Entities"
              value={report.drifted_entities ?? 0}
              className="bg-amber-500/10 text-amber-500"
            />
            <StatCard
              icon={TrendingUp}
              label="Drift Rate"
              value={`${driftPercentage.toFixed(1)}%`}
              className={cn(
                getDriftSeverityColor(driftPercentage),
                driftPercentage <= 5
                  ? 'bg-emerald-500/10'
                  : driftPercentage <= 15
                    ? 'bg-amber-500/10'
                    : driftPercentage <= 30
                      ? 'bg-orange-500/10'
                      : 'bg-red-500/10',
              )}
            />
          </div>

          {/* Progress bar */}
          <Card>
            <CardContent className="p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium">Drift Percentage</h3>
                <div className="flex items-center gap-2">
                  <Badge
                    variant={getDriftSeverityBadgeVariant(driftPercentage)}
                    className="text-[10px]"
                  >
                    {getDriftSeverityLabel(driftPercentage)}
                  </Badge>
                  <span
                    className={cn(
                      'text-sm font-bold tabular-nums',
                      getDriftSeverityColor(driftPercentage),
                    )}
                  >
                    {driftPercentage.toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted">
                <motion.div
                  className={cn(
                    'h-full rounded-full',
                    getDriftProgressColor(driftPercentage),
                  )}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(driftPercentage, 100)}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                />
              </div>

              <p className="mt-2 text-xs text-muted-foreground">
                {report.drifted_entities ?? 0} of{' '}
                {report.total_entities_checked ?? 0} entities have drifted from
                their expected values
              </p>
            </CardContent>
          </Card>

          {/* Drift details */}
          {driftDetails.length > 0 ? (
            <div className="space-y-3">
              <h3 className="text-sm font-medium">
                Drifted Entities ({driftDetails.length})
              </h3>
              {driftDetails.map((detail, idx) => (
                <DriftDetailCard
                  key={detail.entity_id || idx}
                  detail={detail}
                />
              ))}
            </div>
          ) : report.drifted_entities === 0 ? (
            <Card>
              <CardContent className="flex items-center gap-3 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500">
                  <CheckCircle2 className="size-5" />
                </div>
                <div>
                  <h3 className="text-sm font-medium">
                    All entities are consistent
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    No drift detected across checked entities.
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : null}

          {/* Summary */}
          {report.summary && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {report.summary}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

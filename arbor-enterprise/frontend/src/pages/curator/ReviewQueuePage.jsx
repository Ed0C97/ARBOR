import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import {
  ClipboardList,
  Check,
  X,
  Pencil,
  Loader2,
  Inbox,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Database,
  Clock,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { apiGet, apiPost } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getConfidenceColor(confidence) {
  if (confidence >= 0.9) return 'text-emerald-500';
  if (confidence >= 0.7) return 'text-amber-500';
  return 'text-red-500';
}

function getConfidenceBg(confidence) {
  if (confidence >= 0.9) return 'bg-emerald-500/10 border-emerald-500/20';
  if (confidence >= 0.7) return 'bg-amber-500/10 border-amber-500/20';
  return 'bg-red-500/10 border-red-500/20';
}

function formatConfidence(confidence) {
  return `${(confidence * 100).toFixed(0)}%`;
}

function formatDate(dateString) {
  if (!dateString) return '';
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ---------------------------------------------------------------------------
// Proposed Changes Display
// ---------------------------------------------------------------------------

function ProposedChanges({ changes, expanded }) {
  if (!changes) return null;

  const entries = typeof changes === 'object' ? Object.entries(changes) : [];

  if (entries.length === 0) {
    return (
      <p className="text-xs text-muted-foreground italic">No changes specified</p>
    );
  }

  const visibleEntries = expanded ? entries : entries.slice(0, 3);
  const hasMore = entries.length > 3 && !expanded;

  return (
    <div className="space-y-1.5">
      {visibleEntries.map(([key, value]) => (
        <div
          key={key}
          className="flex items-start gap-2 rounded-md bg-muted/50 px-2.5 py-1.5 text-xs"
        >
          <span className="shrink-0 font-medium text-muted-foreground">
            {key}:
          </span>
          <span className="break-all text-foreground">
            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
          </span>
        </div>
      ))}
      {hasMore && (
        <p className="text-[10px] text-muted-foreground">
          +{entries.length - 3} more fields
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review Item Card
// ---------------------------------------------------------------------------

function ReviewItemCard({ item, onApprove, onReject, onOverride, actionLoading }) {
  const [expanded, setExpanded] = useState(false);
  const isLoading = actionLoading === item.id;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6, transition: { duration: 0.15 } }}
      layout
    >
      <Card className="transition-colors hover:bg-accent/20">
        <CardContent className="p-5">
          {/* Top row: entity name + confidence */}
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h3 className="truncate text-sm font-semibold">
                  {item.entity_name}
                </h3>
                <Badge variant="outline" className="shrink-0 text-[10px] px-1.5 py-0 capitalize">
                  {item.enrichment_type?.replace(/_/g, ' ')}
                </Badge>
              </div>

              <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Database className="size-3" />
                  {item.source || 'Unknown source'}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="size-3" />
                  {formatDate(item.created_at)}
                </span>
              </div>
            </div>

            {/* Confidence score */}
            <div
              className={cn(
                'flex shrink-0 flex-col items-center rounded-lg border px-3 py-1.5',
                getConfidenceBg(item.confidence),
              )}
            >
              <span className={cn('text-lg font-bold tabular-nums', getConfidenceColor(item.confidence))}>
                {formatConfidence(item.confidence)}
              </span>
              <span className="text-[10px] text-muted-foreground">confidence</span>
            </div>
          </div>

          {/* Proposed changes */}
          <div className="mt-4">
            <button
              onClick={() => setExpanded((prev) => !prev)}
              className="mb-2 flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              Proposed Changes
              {expanded ? (
                <ChevronUp className="size-3" />
              ) : (
                <ChevronDown className="size-3" />
              )}
            </button>
            <ProposedChanges changes={item.proposed_changes} expanded={expanded} />
          </div>

          <Separator className="my-4" />

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className="border-emerald-500/30 text-emerald-600 hover:bg-emerald-500/10 hover:text-emerald-600"
              disabled={isLoading}
              onClick={() => onApprove(item.id)}
            >
              {isLoading ? (
                <Loader2 className="mr-1 size-3.5 animate-spin" />
              ) : (
                <Check className="mr-1 size-3.5" />
              )}
              Approve
            </Button>

            <Button
              size="sm"
              variant="outline"
              className="border-red-500/30 text-red-600 hover:bg-red-500/10 hover:text-red-600"
              disabled={isLoading}
              onClick={() => onReject(item.id)}
            >
              {isLoading ? (
                <Loader2 className="mr-1 size-3.5 animate-spin" />
              ) : (
                <X className="mr-1 size-3.5" />
              )}
              Reject
            </Button>

            <Button
              size="sm"
              variant="outline"
              className="border-blue-500/30 text-blue-600 hover:bg-blue-500/10 hover:text-blue-600"
              disabled={isLoading}
              onClick={() => onOverride(item)}
            >
              <Pencil className="mr-1 size-3.5" />
              Override
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton Loader
// ---------------------------------------------------------------------------

function QueueSkeleton() {
  return (
    <div className="space-y-4">
      {[1, 2, 3].map((i) => (
        <Card key={i}>
          <CardContent className="p-5 space-y-4">
            <div className="flex items-start justify-between">
              <div className="space-y-2 flex-1">
                <Skeleton className="h-4 w-48" />
                <Skeleton className="h-3 w-32" />
              </div>
              <Skeleton className="h-14 w-16 rounded-lg" />
            </div>
            <div className="space-y-1.5">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-7 w-full" />
              <Skeleton className="h-7 w-3/4" />
            </div>
            <Separator />
            <div className="flex gap-2">
              <Skeleton className="h-8 w-24" />
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-24" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function ReviewQueuePage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Override dialog state
  const [overrideDialogOpen, setOverrideDialogOpen] = useState(false);
  const [overrideItem, setOverrideItem] = useState(null);
  const [overrideData, setOverrideData] = useState('');
  const [overrideNotes, setOverrideNotes] = useState('');

  // -----------------------------------
  // Fetch queue
  // -----------------------------------
  const fetchQueue = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);

      const data = await apiGet('/api/v1/curator/review-queue');
      setItems(Array.isArray(data) ? data : []);
    } catch (err) {
      if (err.name !== 'AbortError') {
        toast.error('Failed to load review queue');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  // -----------------------------------
  // Actions
  // -----------------------------------
  async function handleDecision(itemId, action, overridePayload) {
    try {
      setActionLoading(itemId);

      const body = { action };
      if (overridePayload?.override_data) {
        body.override_data = overridePayload.override_data;
      }
      if (overridePayload?.notes) {
        body.notes = overridePayload.notes;
      }

      await apiPost(`/api/v1/curator/review-queue/${itemId}/decide`, body);

      // Remove item from local state
      setItems((prev) => prev.filter((item) => item.id !== itemId));

      const labels = { approve: 'approved', reject: 'rejected', override: 'overridden' };
      toast.success(`Item ${labels[action]} successfully`);
    } catch (err) {
      toast.error(`Failed to ${action} item`);
    } finally {
      setActionLoading(null);
    }
  }

  function handleApprove(itemId) {
    handleDecision(itemId, 'approve');
  }

  function handleReject(itemId) {
    handleDecision(itemId, 'reject');
  }

  function handleOverrideOpen(item) {
    setOverrideItem(item);
    setOverrideData(
      item.proposed_changes
        ? JSON.stringify(item.proposed_changes, null, 2)
        : '',
    );
    setOverrideNotes('');
    setOverrideDialogOpen(true);
  }

  function handleOverrideSubmit() {
    if (!overrideItem) return;

    let parsedData;
    try {
      parsedData = overrideData ? JSON.parse(overrideData) : undefined;
    } catch {
      toast.error('Override data must be valid JSON');
      return;
    }

    handleDecision(overrideItem.id, 'override', {
      override_data: parsedData,
      notes: overrideNotes || undefined,
    });

    setOverrideDialogOpen(false);
    setOverrideItem(null);
  }

  // -----------------------------------
  // Render
  // -----------------------------------
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <ClipboardList className="size-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold">Review Queue</h2>
              {!loading && (
                <Badge variant="secondary" className="tabular-nums">
                  {items.length}
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              Review and decide on proposed enrichments
            </p>
          </div>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => fetchQueue(true)}
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
        <QueueSkeleton />
      ) : items.length === 0 ? (
        /* Empty state */
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-muted text-muted-foreground">
            <Inbox className="size-8" />
          </div>
          <h3 className="text-sm font-medium">Queue is empty</h3>
          <p className="mt-1 max-w-xs text-xs text-muted-foreground">
            There are no items waiting for review. New enrichments will appear here
            when they are proposed.
          </p>
        </div>
      ) : (
        /* Queue items */
        <div className="space-y-4">
          <AnimatePresence mode="popLayout">
            {items.map((item) => (
              <ReviewItemCard
                key={item.id}
                item={item}
                onApprove={handleApprove}
                onReject={handleReject}
                onOverride={handleOverrideOpen}
                actionLoading={actionLoading}
              />
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Override Dialog */}
      <Dialog open={overrideDialogOpen} onOpenChange={setOverrideDialogOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Override Enrichment</DialogTitle>
            <DialogDescription>
              Provide corrected data for{' '}
              <span className="font-medium text-foreground">
                {overrideItem?.entity_name}
              </span>
              . Edit the JSON below and add optional notes.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="override-data">Override Data (JSON)</Label>
              <Textarea
                id="override-data"
                value={overrideData}
                onChange={(e) => setOverrideData(e.target.value)}
                placeholder='{ "field": "corrected_value" }'
                className="min-h-[140px] font-mono text-xs"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="override-notes">Notes (optional)</Label>
              <Textarea
                id="override-notes"
                value={overrideNotes}
                onChange={(e) => setOverrideNotes(e.target.value)}
                placeholder="Reason for override..."
                className="min-h-[60px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOverrideDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleOverrideSubmit} disabled={!!actionLoading}>
              {actionLoading ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Pencil className="mr-1.5 size-3.5" />
              )}
              Submit Override
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

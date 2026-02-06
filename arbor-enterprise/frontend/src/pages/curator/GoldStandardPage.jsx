import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import {
  Award,
  Plus,
  Trash2,
  Search,
  Loader2,
  RefreshCw,
  FileText,
  AlertTriangle,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { apiGet, apiPost, apiDelete } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FIELD_OPTIONS = [
  { value: 'name', label: 'Name' },
  { value: 'category', label: 'Category' },
  { value: 'description', label: 'Description' },
  { value: 'address', label: 'Address' },
  { value: 'city', label: 'City' },
  { value: 'country', label: 'Country' },
  { value: 'price_range', label: 'Price Range' },
  { value: 'rating', label: 'Rating' },
  { value: 'website', label: 'Website' },
  { value: 'phone', label: 'Phone' },
  { value: 'tags', label: 'Tags' },
  { value: 'hours', label: 'Hours' },
];

const INITIAL_FORM = {
  entity_id: '',
  field: '',
  value: '',
  source: '',
  notes: '',
};

// ---------------------------------------------------------------------------
// Skeleton Loader
// ---------------------------------------------------------------------------

function TableSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center gap-4 px-2 py-3">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-32 flex-1" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-28" />
          <Skeleton className="h-8 w-8" />
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function GoldStandardPage() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Add dialog
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);

  // Delete confirmation
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // -----------------------------------
  // Fetch entries
  // -----------------------------------
  const fetchEntries = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);

      const data = await apiGet('/api/v1/curator/gold-standard');
      setEntries(Array.isArray(data) ? data : []);
    } catch (err) {
      if (err.name !== 'AbortError') {
        toast.error('Failed to load gold standard entries');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  // -----------------------------------
  // Filtered entries
  // -----------------------------------
  const filteredEntries = useMemo(() => {
    if (!searchQuery.trim()) return entries;

    const q = searchQuery.toLowerCase();
    return entries.filter(
      (entry) =>
        entry.entity_id?.toLowerCase().includes(q) ||
        entry.field?.toLowerCase().includes(q) ||
        entry.value?.toLowerCase().includes(q) ||
        entry.source?.toLowerCase().includes(q) ||
        entry.notes?.toLowerCase().includes(q),
    );
  }, [entries, searchQuery]);

  // -----------------------------------
  // Add entry
  // -----------------------------------
  function handleFormChange(field, value) {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }

  async function handleAddSubmit() {
    if (!formData.entity_id || !formData.field || !formData.value) {
      toast.error('Entity ID, Field, and Value are required');
      return;
    }

    try {
      setSubmitting(true);

      const payload = {
        entity_id: formData.entity_id,
        field: formData.field,
        value: formData.value,
        source: formData.source || undefined,
        notes: formData.notes || undefined,
      };

      const created = await apiPost('/api/v1/curator/gold-standard', payload);
      setEntries((prev) => [created, ...prev]);

      toast.success('Gold standard entry created');
      setAddDialogOpen(false);
      setFormData(INITIAL_FORM);
    } catch (err) {
      toast.error('Failed to create gold standard entry');
    } finally {
      setSubmitting(false);
    }
  }

  // -----------------------------------
  // Delete entry
  // -----------------------------------
  function handleDeleteOpen(entry) {
    setDeleteTarget(entry);
    setDeleteDialogOpen(true);
  }

  async function handleDeleteConfirm() {
    if (!deleteTarget) return;

    try {
      setDeleting(true);
      await apiDelete(`/api/v1/curator/gold-standard/${deleteTarget.id}`);

      setEntries((prev) => prev.filter((e) => e.id !== deleteTarget.id));
      toast.success('Entry deleted');
      setDeleteDialogOpen(false);
      setDeleteTarget(null);
    } catch (err) {
      toast.error('Failed to delete entry');
    } finally {
      setDeleting(false);
    }
  }

  // -----------------------------------
  // Render
  // -----------------------------------
  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-500/10 text-amber-500">
            <Award className="size-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold">Gold Standard</h2>
              {!loading && (
                <Badge variant="secondary" className="tabular-nums">
                  {entries.length}
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              Curated ground-truth values for entity fields
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchEntries(true)}
            disabled={refreshing}
          >
            <RefreshCw
              className={cn('mr-1.5 size-3.5', refreshing && 'animate-spin')}
            />
            Refresh
          </Button>
          <Button
            size="sm"
            onClick={() => {
              setFormData(INITIAL_FORM);
              setAddDialogOpen(true);
            }}
          >
            <Plus className="mr-1.5 size-3.5" />
            Add New
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search entries..."
          className="pl-9"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <Separator />

      {/* Table */}
      {loading ? (
        <TableSkeleton />
      ) : filteredEntries.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-muted text-muted-foreground">
            <FileText className="size-8" />
          </div>
          <h3 className="text-sm font-medium">
            {searchQuery ? 'No matching entries' : 'No gold standard entries'}
          </h3>
          <p className="mt-1 max-w-xs text-xs text-muted-foreground">
            {searchQuery
              ? 'Try adjusting your search query'
              : 'Create your first gold standard entry to establish ground-truth data.'}
          </p>
          {!searchQuery && (
            <Button
              size="sm"
              className="mt-4"
              onClick={() => {
                setFormData(INITIAL_FORM);
                setAddDialogOpen(true);
              }}
            >
              <Plus className="mr-1.5 size-3.5" />
              Add First Entry
            </Button>
          )}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Entity</TableHead>
                <TableHead>Field</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Notes</TableHead>
                <TableHead className="w-[60px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEntries.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell className="font-medium">
                    {entry.entity_id}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0 capitalize">
                      {entry.field}
                    </Badge>
                  </TableCell>
                  <TableCell className="max-w-[200px] truncate">
                    {entry.value}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {entry.source || '--'}
                  </TableCell>
                  <TableCell className="max-w-[160px] truncate text-muted-foreground">
                    {entry.notes || '--'}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8 text-muted-foreground hover:text-red-600"
                      onClick={() => handleDeleteOpen(entry)}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {searchQuery && (
            <p className="mt-3 text-xs text-muted-foreground">
              Showing {filteredEntries.length} of {entries.length} entries
            </p>
          )}
        </motion.div>
      )}

      {/* Add Entry Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add Gold Standard Entry</DialogTitle>
            <DialogDescription>
              Define a ground-truth value for an entity field.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="gs-entity-id">
                Entity ID <span className="text-red-500">*</span>
              </Label>
              <Input
                id="gs-entity-id"
                value={formData.entity_id}
                onChange={(e) => handleFormChange('entity_id', e.target.value)}
                placeholder="e.g. ent_abc123"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="gs-field">
                Field <span className="text-red-500">*</span>
              </Label>
              <Select
                value={formData.field}
                onValueChange={(v) => handleFormChange('field', v)}
              >
                <SelectTrigger id="gs-field">
                  <SelectValue placeholder="Select a field" />
                </SelectTrigger>
                <SelectContent>
                  {FIELD_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="gs-value">
                Value <span className="text-red-500">*</span>
              </Label>
              <Textarea
                id="gs-value"
                value={formData.value}
                onChange={(e) => handleFormChange('value', e.target.value)}
                placeholder="The correct value for this field"
                className="min-h-[80px]"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="gs-source">Source</Label>
              <Input
                id="gs-source"
                value={formData.source}
                onChange={(e) => handleFormChange('source', e.target.value)}
                placeholder="e.g. official website, manual research"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="gs-notes">Notes</Label>
              <Textarea
                id="gs-notes"
                value={formData.notes}
                onChange={(e) => handleFormChange('notes', e.target.value)}
                placeholder="Additional context or reasoning..."
                className="min-h-[60px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAddDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleAddSubmit} disabled={submitting}>
              {submitting ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Plus className="mr-1.5 size-3.5" />
              )}
              Create Entry
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="size-5 text-red-500" />
              Confirm Deletion
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the gold standard entry for{' '}
              <span className="font-medium text-foreground">
                {deleteTarget?.field}
              </span>{' '}
              on entity{' '}
              <span className="font-medium text-foreground">
                {deleteTarget?.entity_id}
              </span>
              ? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleting}
            >
              {deleting ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Trash2 className="mr-1.5 size-3.5" />
              )}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

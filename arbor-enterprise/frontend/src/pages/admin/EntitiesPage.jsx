import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Search,
  Store,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Eye,
  Sparkles,
  Trash2,
  AlertCircle,
  RefreshCw,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { toast } from 'sonner';

import { cn, formatNumber } from '@/lib/utils';
import { useAdminStore } from '@/stores/adminStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

const PAGE_SIZE = 20;

export default function AdminEntitiesPage() {
  const {
    entities,
    entitiesTotal,
    entitiesLoading,
    entitiesError,
    fetchEntities,
    enrichEntity,
    cancelEnrichment,
  } = useAdminStore();

  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [page, setPage] = useState(1);
  const [enrichingIds, setEnrichingIds] = useState(new Set());
  const [cancellingIds, setCancellingIds] = useState(new Set());

  const loadEntities = useCallback(() => {
    const params = {
      limit: PAGE_SIZE,
      offset: (page - 1) * PAGE_SIZE,
    };
    if (search) params.search = search;
    if (typeFilter !== 'all') params.entity_type = typeFilter;
    fetchEntities(params);
  }, [page, search, typeFilter, fetchEntities]);

  useEffect(() => {
    loadEntities();
  }, [loadEntities]);

  // Debounced search: reset page on filter change
  useEffect(() => {
    setPage(1);
  }, [search, typeFilter]);

  const totalPages = Math.ceil(entitiesTotal / PAGE_SIZE);

  async function handleEnrich(entityId) {
    setEnrichingIds((prev) => new Set(prev).add(entityId));
    try {
      await enrichEntity(entityId);
      toast.success('Enrichment triggered successfully');
      loadEntities();
    } catch (err) {
      toast.error(`Enrichment failed: ${err.message}`);
    } finally {
      setEnrichingIds((prev) => {
        const next = new Set(prev);
        next.delete(entityId);
        return next;
      });
    }
  }

  async function handleCancelEnrichment(entityId) {
    setCancellingIds((prev) => new Set(prev).add(entityId));
    try {
      await cancelEnrichment(entityId);
      toast.success('Enrichment cancelled');
      loadEntities();
    } catch (err) {
      toast.error(`Cancel failed: ${err.message}`);
    } finally {
      setCancellingIds((prev) => {
        const next = new Set(prev);
        next.delete(entityId);
        return next;
      });
    }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Store className="size-5" />
            Entity Management
          </h2>
          <p className="text-sm text-muted-foreground">
            {entitiesTotal > 0
              ? `${formatNumber(entitiesTotal)} entities total`
              : 'Manage all entities in the system'}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={loadEntities}
          disabled={entitiesLoading}
        >
          <RefreshCw className={cn('size-4', entitiesLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      <Separator />

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-[200px] max-w-sm flex-1">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search entities..."
            className="pl-9"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <Select
          value={typeFilter}
          onValueChange={(v) => setTypeFilter(v)}
        >
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="brand">Brand</SelectItem>
            <SelectItem value="venue">Venue</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Error State */}
      {entitiesError && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-center gap-3 p-4">
            <AlertCircle className="size-5 text-destructive" />
            <div>
              <p className="text-sm font-medium">Failed to load entities</p>
              <p className="text-xs text-muted-foreground">{entitiesError}</p>
            </div>
            <Button variant="outline" size="sm" className="ml-auto" onClick={loadEntities}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Table */}
      <Card>
        <CardContent className="p-0">
          {entitiesLoading && entities.length === 0 ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="size-6 animate-spin text-muted-foreground" />
            </div>
          ) : entities.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <Store className="mb-4 size-10 text-muted-foreground/50" />
              <h3 className="text-sm font-medium">No entities found</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Try adjusting your search or filters
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="hidden sm:table-cell">Category</TableHead>
                  <TableHead className="hidden md:table-cell">City</TableHead>
                  <TableHead className="hidden lg:table-cell">Rating</TableHead>
                  <TableHead>Verified</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {entities.map((entity) => (
                  <TableRow key={entity.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                          <Store className="size-4" />
                        </div>
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">{entity.name}</p>
                          {entity.description && (
                            <p className="truncate text-xs text-muted-foreground max-w-[200px]">
                              {entity.description}
                            </p>
                          )}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="capitalize text-xs">
                        {entity.entity_type || 'unknown'}
                      </Badge>
                    </TableCell>
                    <TableCell className="hidden sm:table-cell">
                      <span className="text-sm text-muted-foreground">
                        {entity.category || '--'}
                      </span>
                    </TableCell>
                    <TableCell className="hidden md:table-cell">
                      <span className="text-sm text-muted-foreground">
                        {entity.city || '--'}
                      </span>
                    </TableCell>
                    <TableCell className="hidden lg:table-cell">
                      <span className="text-sm">
                        {entity.rating ? entity.rating.toFixed(1) : '--'}
                      </span>
                    </TableCell>
                    <TableCell>
                      {entity.verified ? (
                        <CheckCircle2 className="size-4 text-emerald-500" />
                      ) : (
                        <XCircle className="size-4 text-muted-foreground/40" />
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        <Link to={`/entity/${entity.id}`}>
                          <Button variant="ghost" size="icon" className="size-8">
                            <Eye className="size-4" />
                          </Button>
                        </Link>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-8 text-primary hover:text-primary"
                          onClick={() => handleEnrich(entity.id)}
                          disabled={enrichingIds.has(entity.id)}
                        >
                          {enrichingIds.has(entity.id) ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            <Sparkles className="size-4" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-8 text-destructive hover:text-destructive"
                          onClick={() => handleCancelEnrichment(entity.id)}
                          disabled={cancellingIds.has(entity.id)}
                        >
                          {cancellingIds.has(entity.id) ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            <Trash2 className="size-4" />
                          )}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft className="mr-1 size-4" />
            Previous
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
            <ChevronRight className="ml-1 size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

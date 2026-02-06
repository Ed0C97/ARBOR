import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Search,
  Filter,
  MapPin,
  Star,
  Store,
  Tag,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Grid3X3,
  List,
} from 'lucide-react';
import { motion } from 'framer-motion';

import { cn } from '@/lib/utils';
import { useEntityStore } from '@/stores/entityStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';

function EntityCard({ entity, viewMode }) {
  const isGrid = viewMode === 'grid';

  return (
    <Link to={`/entity/${entity.id}`}>
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="group cursor-pointer transition-colors hover:bg-accent/30">
          <CardContent className={cn('p-4', !isGrid && 'flex items-center gap-4')}>
            {/* Icon */}
            <div
              className={cn(
                'flex shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary',
                isGrid ? 'mb-3 h-12 w-12' : 'h-10 w-10',
              )}
            >
              <Store className="size-5" />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-2">
                <h3 className="truncate text-sm font-semibold group-hover:text-primary">
                  {entity.name}
                </h3>
                {entity.rating && (
                  <span className="flex shrink-0 items-center gap-0.5 text-xs font-medium text-amber-500">
                    <Star className="size-3 fill-current" />
                    {entity.rating}
                  </span>
                )}
              </div>

              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                {entity.entity_type && (
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0 capitalize">
                    {entity.entity_type}
                  </Badge>
                )}
                {entity.category && (
                  <span className="flex items-center gap-0.5">
                    <Tag className="size-3" />
                    {entity.category}
                  </span>
                )}
                {entity.city && (
                  <span className="flex items-center gap-0.5">
                    <MapPin className="size-3" />
                    {entity.city}{entity.country ? `, ${entity.country}` : ''}
                  </span>
                )}
              </div>

              {isGrid && entity.description && (
                <p className="mt-2 line-clamp-2 text-xs text-muted-foreground">
                  {entity.description}
                </p>
              )}
            </div>

            {!isGrid && entity.price_range && (
              <span className="shrink-0 text-xs text-muted-foreground">
                {entity.price_range}
              </span>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </Link>
  );
}

export default function BrowsePage() {
  const [viewMode, setViewMode] = useState('grid');
  const {
    entities,
    total,
    loading,
    filters,
    setFilter,
    fetchEntities,
    nextPage,
    prevPage,
    hasMore,
    currentPage,
  } = useEntityStore();

  useEffect(() => {
    fetchEntities();
  }, [fetchEntities]);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Header / Filters */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Browse Entities</h2>
          <p className="text-sm text-muted-foreground">
            {total > 0 ? `${total} entities found` : 'Loading...'}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
            size="icon"
            className="size-8"
            onClick={() => setViewMode('grid')}
          >
            <Grid3X3 className="size-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'secondary' : 'ghost'}
            size="icon"
            className="size-8"
            onClick={() => setViewMode('list')}
          >
            <List className="size-4" />
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search entities..."
            className="pl-9"
            value={filters.search || ''}
            onChange={(e) => setFilter('search', e.target.value)}
          />
        </div>

        <Select
          value={filters.type || 'all'}
          onValueChange={(v) => setFilter('type', v === 'all' ? '' : v)}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="brand">Brands</SelectItem>
            <SelectItem value="venue">Venues</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={filters.category || 'all'}
          onValueChange={(v) => setFilter('category', v === 'all' ? '' : v)}
        >
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            <SelectItem value="tailoring">Tailoring</SelectItem>
            <SelectItem value="menswear">Menswear</SelectItem>
            <SelectItem value="accessories">Accessories</SelectItem>
            <SelectItem value="footwear">Footwear</SelectItem>
            <SelectItem value="vintage">Vintage</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Separator />

      {/* Entity Grid/List */}
      {loading ? (
        <div className="flex items-center justify-center py-24">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      ) : entities.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Store className="mb-4 size-10 text-muted-foreground/50" />
          <h3 className="text-sm font-medium">No entities found</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Try adjusting your filters or search query
          </p>
        </div>
      ) : (
        <>
          <div
            className={cn(
              viewMode === 'grid'
                ? 'grid gap-4 sm:grid-cols-2 lg:grid-cols-3'
                : 'space-y-2',
            )}
          >
            {entities.map((entity) => (
              <EntityCard key={entity.id} entity={entity} viewMode={viewMode} />
            ))}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-4">
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage <= 1}
              onClick={prevPage}
            >
              <ChevronLeft className="mr-1 size-4" />
              Previous
            </Button>
            <span className="text-xs text-muted-foreground">
              Page {currentPage}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasMore}
              onClick={nextPage}
            >
              Next
              <ChevronRight className="ml-1 size-4" />
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

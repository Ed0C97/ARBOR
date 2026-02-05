"use client";

import Link from "next/link";
import { MapPin, Tag, BadgeCheck, ArrowUpRight, Star, Instagram } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Entity } from "@/lib/types";

interface EntityCardProps {
  entity: Entity;
  className?: string;
}

export function EntityCard({ entity, className }: EntityCardProps) {
  return (
    <Link href={`/entity/${entity.id}`}>
      <Card
        className={cn(
          "group cursor-pointer transition-all hover:border-primary/30 hover:shadow-glow",
          className,
        )}
      >
        <CardContent className="p-5">
          <div className="space-y-3">
            {/* Header */}
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="truncate font-serif text-base font-semibold group-hover:text-primary">
                    {entity.name}
                  </h3>
                  {entity.verified && (
                    <BadgeCheck className="h-4 w-4 shrink-0 text-primary" />
                  )}
                  {entity.is_featured && (
                    <Star className="h-3.5 w-3.5 shrink-0 text-yellow-400" />
                  )}
                </div>

                {/* Category, City & Instagram */}
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  {entity.category && (
                    <span className="flex items-center gap-1">
                      <Tag className="h-3 w-3" />
                      {entity.category}
                    </span>
                  )}
                  {entity.city && (
                    <span className="flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      {entity.city}
                    </span>
                  )}
                  {entity.instagram && (
                    <span className="flex items-center gap-1">
                      <Instagram className="h-3 w-3" />
                      @{entity.instagram.replace("@", "")}
                    </span>
                  )}
                </div>
              </div>

              {/* Price range & type badge */}
              <div className="flex shrink-0 flex-col items-end gap-1">
                {entity.price_range && (
                  <span className="bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">
                    {entity.price_range}
                  </span>
                )}
                <span
                  className={cn(
                    "px-2 py-0.5 font-mono uppercase tracking-wider text-[10px] font-medium",
                    entity.entity_type === "brand"
                      ? "bg-arbor-tobacco/20 text-arbor-camel"
                      : "bg-primary/20 text-primary",
                  )}
                >
                  {entity.entity_type}
                </span>
              </div>
            </div>

            {/* Description */}
            {entity.description && (
              <p className="line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                {entity.description}
              </p>
            )}

            {/* Tags from enrichment */}
            {entity.tags && entity.tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {entity.tags.slice(0, 5).map((tag) => (
                  <span
                    key={tag}
                    className="bg-primary/10 px-2.5 py-0.5 text-[11px] font-mono uppercase font-medium text-primary"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* Footer */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-2">
                {entity.is_active ? (
                  <span className="bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                    Active
                  </span>
                ) : (
                  <span className="bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                    Inactive
                  </span>
                )}
                {entity.rating && (
                  <span className="flex items-center gap-0.5 font-mono text-xs text-muted-foreground">
                    <Star className="h-3 w-3 text-yellow-400" />
                    {entity.rating.toFixed(1)}
                  </span>
                )}
              </div>
              <span className="flex items-center gap-1 font-mono text-xs font-medium text-muted-foreground transition-colors group-hover:text-primary">
                View
                <ArrowUpRight className="h-3 w-3" />
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

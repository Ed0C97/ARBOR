"use client";

import {
  MapPin,
  Phone,
  Globe,
  Tag,
  BadgeCheck,
  Clock,
  Sparkles,
  Users,
  Star,
  Instagram,
  Mail,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { VibeRadar } from "./VibeRadar";
import { cn } from "@/lib/utils";
import type { Entity, VibeDNA } from "@/lib/types";

interface EntityDetailProps {
  entity: Entity;
}

export function EntityDetail({ entity }: EntityDetailProps) {
  // vibe_dna may come from the enrichment as a generic dict
  const vibeDna = entity.vibe_dna as VibeDNA | null;
  const hasDimensions =
    vibeDna?.dimensions && Object.keys(vibeDna.dimensions).length > 0;

  return (
    <div className="space-y-6">
      {/* ---------- Header card ---------- */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
            {/* Name & meta */}
            <div className="min-w-0 flex-1 space-y-3">
              <div className="flex items-center gap-3">
                <h1 className="font-serif text-2xl tracking-tight md:text-3xl">
                  {entity.name}
                </h1>
                {entity.verified && (
                  <BadgeCheck className="h-5 w-5 shrink-0 text-primary" />
                )}
                {entity.is_featured && (
                  <Star className="h-5 w-5 shrink-0 text-yellow-400" />
                )}
              </div>

              {/* Meta row */}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
                <span
                  className={cn(
                    "px-2.5 py-0.5 text-xs font-mono uppercase tracking-wider font-medium",
                    entity.entity_type === "brand"
                      ? "bg-arbor-tobacco/20 text-arbor-camel"
                      : "bg-primary/20 text-primary",
                  )}
                >
                  {entity.entity_type}
                </span>
                {entity.category && (
                  <span className="flex items-center gap-1.5">
                    <Tag className="h-3.5 w-3.5" />
                    {entity.category}
                  </span>
                )}
                {entity.city && (
                  <span className="flex items-center gap-1.5">
                    <MapPin className="h-3.5 w-3.5" />
                    {entity.city}
                    {entity.country ? `, ${entity.country}` : ""}
                  </span>
                )}
                {entity.price_range && (
                  <span className="bg-muted px-2 py-0.5 text-xs font-medium">
                    {entity.price_range}
                  </span>
                )}
                {entity.rating && (
                  <span className="flex items-center gap-1 font-mono">
                    <Star className="h-3.5 w-3.5 text-yellow-400" />
                    {entity.rating.toFixed(1)}
                  </span>
                )}
                {entity.is_active ? (
                  <span className="bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                    Active
                  </span>
                ) : (
                  <span className="bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                    Inactive
                  </span>
                )}
              </div>

              {/* Style & Gender */}
              {(entity.style || entity.gender) && (
                <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
                  {entity.style && (
                    <span className="bg-muted px-2 py-0.5 text-xs">
                      Style: {entity.style}
                    </span>
                  )}
                  {entity.gender && (
                    <span className="bg-muted px-2 py-0.5 text-xs">
                      {entity.gender}
                    </span>
                  )}
                </div>
              )}

              {/* Description */}
              {entity.description && (
                <p className="max-w-2xl leading-relaxed text-muted-foreground">
                  {entity.description}
                </p>
              )}

              {/* Specialty */}
              {entity.specialty && (
                <p className="text-sm text-muted-foreground">
                  <strong className="text-foreground">Specialty:</strong>{" "}
                  {entity.specialty}
                </p>
              )}
            </div>

            {/* Contact info */}
            <div className="flex shrink-0 flex-col gap-2 text-sm">
              {entity.address && (
                <span className="flex items-center gap-2 text-muted-foreground">
                  <MapPin className="h-4 w-4 shrink-0" />
                  {entity.address}
                </span>
              )}
              {entity.phone && (
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Phone className="h-4 w-4 shrink-0" />
                  {entity.phone}
                </span>
              )}
              {entity.email && (
                <a
                  href={`mailto:${entity.email}`}
                  className="flex items-center gap-2 text-primary transition-colors hover:text-primary/80"
                >
                  <Mail className="h-4 w-4 shrink-0" />
                  {entity.email}
                </a>
              )}
              {entity.website && (
                <a
                  href={entity.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-primary transition-colors hover:text-primary/80"
                >
                  <Globe className="h-4 w-4 shrink-0" />
                  Website
                </a>
              )}
              {entity.instagram && (
                <a
                  href={`https://instagram.com/${entity.instagram.replace("@", "")}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-primary transition-colors hover:text-primary/80"
                >
                  <Instagram className="h-4 w-4 shrink-0" />
                  @{entity.instagram.replace("@", "")}
                </a>
              )}
              {entity.maps_url && (
                <a
                  href={entity.maps_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-primary transition-colors hover:text-primary/80"
                >
                  <ExternalLink className="h-4 w-4 shrink-0" />
                  Maps
                </a>
              )}
              {entity.created_at && (
                <span className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="h-3.5 w-3.5 shrink-0" />
                  Added {new Date(entity.created_at).toLocaleDateString()}
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ---------- ARBOR Enrichment Tags ---------- */}
      {entity.tags && entity.tags.length > 0 && !vibeDna && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-serif text-lg">
              <Tag className="h-5 w-5 text-primary" />
              ARBOR Tags
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {entity.tags.map((tag) => (
                <span
                  key={tag}
                  className="border border-border bg-muted/50 px-3 py-1 text-sm font-mono text-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ---------- Vibe DNA section ---------- */}
      {vibeDna && (
        <div className="grid gap-6 md:grid-cols-2">
          {/* Radar chart */}
          {hasDimensions && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 font-serif text-lg">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Vibe DNA
                </CardTitle>
              </CardHeader>
              <CardContent>
                <VibeRadar
                  dimensions={vibeDna.dimensions}
                  className="mx-auto max-w-sm"
                />
              </CardContent>
            </Card>
          )}

          {/* Vibe details */}
          <div className="space-y-6">
            {/* Tags */}
            {vibeDna.tags && vibeDna.tags.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-serif text-lg">
                    <Tag className="h-5 w-5 text-primary" />
                    Tags
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {vibeDna.tags.map((tag) => (
                      <span
                        key={tag}
                        className="border border-border bg-muted/50 px-3 py-1 text-sm font-mono text-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Signature items */}
            {vibeDna.signature_items && vibeDna.signature_items.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-serif text-lg">
                    <Star className="h-5 w-5 text-primary" />
                    Signature Items
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {vibeDna.signature_items.map((item) => (
                      <li
                        key={item}
                        className="flex items-center gap-2 text-sm text-muted-foreground"
                      >
                        <div className="h-1.5 w-1.5 shrink-0 bg-primary" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* Target audience */}
            {vibeDna.target_audience && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-serif text-lg">
                    <Users className="h-5 w-5 text-primary" />
                    Target Audience
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {vibeDna.target_audience}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Vibe summary */}
            {vibeDna.summary && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 font-serif text-lg">
                    <Sparkles className="h-5 w-5 text-primary" />
                    Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {vibeDna.summary}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* ---------- Notes ---------- */}
      {entity.notes && (
        <Card>
          <CardHeader>
            <CardTitle className="font-serif text-lg">Notes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {entity.notes}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

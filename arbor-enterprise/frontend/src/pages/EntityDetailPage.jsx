import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  MapPin,
  Globe,
  Instagram,
  Mail,
  Phone,
  Star,
  Tag,
  Network,
  ExternalLink,
  Loader2,
  CheckCircle2,
  User,
} from 'lucide-react';

import { apiGet } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

function InfoRow({ icon: Icon, label, value, href }) {
  if (!value) return null;

  const content = (
    <div className="flex items-center gap-3 py-2">
      <Icon className="size-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="truncate text-sm">{value}</p>
      </div>
      {href && <ExternalLink className="size-3 shrink-0 text-muted-foreground" />}
    </div>
  );

  if (href) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="block hover:bg-accent/30 rounded-md px-2 -mx-2 transition-colors">
        {content}
      </a>
    );
  }

  return <div className="px-2 -mx-2">{content}</div>;
}

export default function EntityDetailPage() {
  const { id } = useParams();
  const [entity, setEntity] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await apiGet(`/api/v1/entities/${id}`);
        setEntity(data);
      } catch (err) {
        setError(err.message || 'Failed to load entity');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !entity) {
    return (
      <div className="flex flex-col items-center justify-center py-32 text-center">
        <h3 className="text-sm font-medium">Entity not found</h3>
        <p className="mt-1 text-xs text-muted-foreground">{error}</p>
        <Button asChild variant="outline" size="sm" className="mt-4">
          <Link to="/browse">
            <ArrowLeft className="mr-1 size-4" />
            Back to Browse
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      {/* Back button */}
      <Button asChild variant="ghost" size="sm" className="-ml-2">
        <Link to="/browse">
          <ArrowLeft className="mr-1 size-4" />
          Back
        </Link>
      </Button>

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold">{entity.name}</h1>
            {entity.verified && (
              <CheckCircle2 className="size-5 text-primary" />
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {entity.entity_type && (
              <Badge variant="secondary" className="capitalize">
                {entity.entity_type}
              </Badge>
            )}
            {entity.category && (
              <Badge variant="outline">{entity.category}</Badge>
            )}
            {entity.is_featured && (
              <Badge className="bg-amber-500/10 text-amber-600 border-amber-500/20">
                Featured
              </Badge>
            )}
            {entity.tags?.map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {entity.rating && (
            <div className="flex items-center gap-1 rounded-lg bg-amber-500/10 px-3 py-1.5">
              <Star className="size-4 fill-amber-500 text-amber-500" />
              <span className="text-sm font-semibold text-amber-600">
                {entity.rating}
              </span>
            </div>
          )}
          <Button asChild variant="outline" size="sm">
            <Link to={`/graph?entity=${entity.id}`}>
              <Network className="mr-1 size-4" />
              View in Graph
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main content */}
        <div className="space-y-6 lg:col-span-2">
          {entity.description && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">About</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {entity.description}
                </p>
              </CardContent>
            </Card>
          )}

          {entity.specialty && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Specialty</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{entity.specialty}</p>
              </CardContent>
            </Card>
          )}

          {entity.notes && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">{entity.notes}</p>
              </CardContent>
            </Card>
          )}

          {entity.vibe_dna && Object.keys(entity.vibe_dna).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Vibe DNA</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(entity.vibe_dna).map(([key, value]) => (
                    <div key={key}>
                      <div className="mb-1 flex items-center justify-between text-xs">
                        <span className="capitalize text-muted-foreground">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <span className="font-medium">
                          {typeof value === 'number' ? `${Math.round(value * 100)}%` : value}
                        </span>
                      </div>
                      {typeof value === 'number' && (
                        <div className="h-1.5 w-full rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary transition-all"
                            style={{ width: `${Math.round(value * 100)}%` }}
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar info */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Contact & Location</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              <InfoRow icon={MapPin} label="Address" value={entity.address} href={entity.maps_url} />
              <InfoRow icon={MapPin} label="City" value={[entity.city, entity.region, entity.country].filter(Boolean).join(', ')} />
              <Separator className="my-2" />
              <InfoRow icon={Globe} label="Website" value={entity.website} href={entity.website} />
              <InfoRow icon={Instagram} label="Instagram" value={entity.instagram} href={entity.instagram?.startsWith('http') ? entity.instagram : `https://instagram.com/${entity.instagram?.replace('@', '')}`} />
              <InfoRow icon={Mail} label="Email" value={entity.email} href={entity.email ? `mailto:${entity.email}` : null} />
              <InfoRow icon={Phone} label="Phone" value={entity.phone} href={entity.phone ? `tel:${entity.phone}` : null} />
              <InfoRow icon={User} label="Contact Person" value={entity.contact_person} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {entity.price_range && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Price Range</span>
                  <span className="font-medium">{entity.price_range}</span>
                </div>
              )}
              {entity.style && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Style</span>
                  <span className="font-medium capitalize">{entity.style}</span>
                </div>
              )}
              {entity.gender && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Gender</span>
                  <span className="font-medium capitalize">{entity.gender}</span>
                </div>
              )}
              {entity.priority !== undefined && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Priority</span>
                  <span className="font-medium">{entity.priority}</span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

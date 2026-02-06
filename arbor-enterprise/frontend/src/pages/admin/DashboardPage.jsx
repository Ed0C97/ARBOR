import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LayoutDashboard,
  Store,
  Building2,
  MapPin,
  Sparkles,
  Activity,
  Server,
  Clock,
  ArrowUpRight,
  Loader2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

import { cn, formatNumber } from '@/lib/utils';
import { useAdminStore } from '@/stores/adminStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';

// Placeholder metric data for the area chart when real metrics are unavailable
const PLACEHOLDER_METRICS = [
  { name: 'Mon', requests: 120, latency: 45 },
  { name: 'Tue', requests: 180, latency: 38 },
  { name: 'Wed', requests: 250, latency: 42 },
  { name: 'Thu', requests: 210, latency: 51 },
  { name: 'Fri', requests: 340, latency: 36 },
  { name: 'Sat', requests: 160, latency: 29 },
  { name: 'Sun', requests: 90, latency: 33 },
];

function KpiCard({ title, value, icon: Icon, description, loading }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {title}
          </CardTitle>
          <Icon className="size-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-8 w-24" />
          ) : (
            <>
              <div className="text-2xl font-bold">{value}</div>
              {description && (
                <p className="mt-1 text-xs text-muted-foreground">{description}</p>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function RecentActivityPlaceholder() {
  const items = [
    { action: 'Entity enriched', target: 'Savile Row Tailors', time: '2 min ago', type: 'success' },
    { action: 'New entity added', target: 'Barker Shoes', time: '15 min ago', type: 'default' },
    { action: 'Enrichment failed', target: 'Milano Vintage', time: '1 hr ago', type: 'destructive' },
    { action: 'Entity updated', target: 'Anderson & Sheppard', time: '2 hr ago', type: 'default' },
    { action: 'Batch enrichment started', target: '12 entities', time: '3 hr ago', type: 'warning' },
  ];

  return (
    <div className="space-y-3">
      {items.map((item, i) => (
        <div key={i} className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className={cn(
              'size-2 shrink-0 rounded-full',
              item.type === 'success' && 'bg-emerald-500',
              item.type === 'destructive' && 'bg-destructive',
              item.type === 'warning' && 'bg-amber-500',
              item.type === 'default' && 'bg-primary',
            )} />
            <div className="min-w-0">
              <p className="truncate text-sm">{item.action}</p>
              <p className="truncate text-xs text-muted-foreground">{item.target}</p>
            </div>
          </div>
          <span className="shrink-0 text-xs text-muted-foreground">{item.time}</span>
        </div>
      ))}
    </div>
  );
}

export default function AdminDashboardPage() {
  const {
    stats,
    statsLoading,
    statsError,
    health,
    healthLoading,
    metrics,
    metricsLoading,
    fetchStats,
    fetchHealth,
    fetchMetrics,
  } = useAdminStore();

  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchStats();
    fetchHealth();
    fetchMetrics();
  }, [fetchStats, fetchHealth, fetchMetrics]);

  async function handleRefresh() {
    setRefreshing(true);
    try {
      await Promise.all([fetchStats(), fetchHealth(), fetchMetrics()]);
    } finally {
      setRefreshing(false);
    }
  }

  // Build chart data from metrics or use placeholder
  const chartData = metrics?.chart_data || metrics?.data || PLACEHOLDER_METRICS;

  const kpis = [
    {
      title: 'Total Entities',
      value: formatNumber(stats?.total_entities ?? 0),
      icon: Store,
      description: 'Across all types',
    },
    {
      title: 'Brands',
      value: formatNumber(stats?.total_brands ?? 0),
      icon: Building2,
      description: 'Brand entities',
    },
    {
      title: 'Venues',
      value: formatNumber(stats?.total_venues ?? 0),
      icon: MapPin,
      description: 'Venue entities',
    },
    {
      title: 'Enriched',
      value: formatNumber(stats?.enriched_entities ?? 0),
      icon: Sparkles,
      description: stats?.total_entities
        ? `${Math.round((stats.enriched_entities / stats.total_entities) * 100)}% complete`
        : 'Enrichment progress',
    },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <LayoutDashboard className="size-5" />
            Admin Dashboard
          </h2>
          <p className="text-sm text-muted-foreground">
            System overview and key metrics
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw className={cn('size-4', refreshing && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      <Separator />

      {/* Error State */}
      {statsError && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-center gap-3 p-4">
            <AlertCircle className="size-5 text-destructive" />
            <div>
              <p className="text-sm font-medium">Failed to load stats</p>
              <p className="text-xs text-muted-foreground">{statsError}</p>
            </div>
            <Button variant="outline" size="sm" className="ml-auto" onClick={fetchStats}>
              Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {/* KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {kpis.map((kpi) => (
          <KpiCard
            key={kpi.title}
            title={kpi.title}
            value={kpi.value}
            icon={kpi.icon}
            description={kpi.description}
            loading={statsLoading}
          />
        ))}
      </div>

      {/* Charts & Activity Row */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Area Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Request Volume</CardTitle>
            <CardDescription>API requests over the past week</CardDescription>
          </CardHeader>
          <CardContent>
            {metricsLoading ? (
              <Skeleton className="h-[240px] w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorRequests" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                    axisLine={{ stroke: 'hsl(var(--border))' }}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                    axisLine={{ stroke: 'hsl(var(--border))' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="requests"
                    stroke="hsl(var(--primary))"
                    fillOpacity={1}
                    fill="url(#colorRequests)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Recent Activity</CardTitle>
              <Activity className="size-4 text-muted-foreground" />
            </div>
            <CardDescription>Latest system events</CardDescription>
          </CardHeader>
          <CardContent>
            <RecentActivityPlaceholder />
          </CardContent>
        </Card>
      </div>

      {/* System Status Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-medium">System Status</CardTitle>
              <CardDescription>Health overview from readiness endpoint</CardDescription>
            </div>
            <Link to="/admin/health">
              <Button variant="ghost" size="sm">
                View Details
                <ArrowUpRight className="ml-1 size-3" />
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <div className="flex items-center gap-3">
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Checking system health...</span>
            </div>
          ) : health ? (
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Status</p>
                <Badge variant={health.status === 'healthy' ? 'success' : health.status === 'degraded' ? 'warning' : 'destructive'}>
                  {health.status}
                </Badge>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Version</p>
                <p className="text-sm font-medium">{health.version || 'N/A'}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Environment</p>
                <p className="text-sm font-medium capitalize">{health.environment || 'N/A'}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Unable to fetch system status</p>
          )}
        </CardContent>
      </Card>

      {/* Quick Links */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: 'Manage Entities', to: '/admin/entities', icon: Store },
          { label: 'Data Ingestion', to: '/admin/ingestion', icon: Sparkles },
          { label: 'Analytics', to: '/admin/analytics', icon: Activity },
          { label: 'System Health', to: '/admin/health', icon: Server },
        ].map((link) => (
          <Link key={link.to} to={link.to}>
            <Card className="group cursor-pointer transition-colors hover:bg-accent/30">
              <CardContent className="flex items-center gap-3 p-4">
                <link.icon className="size-5 text-muted-foreground group-hover:text-primary" />
                <span className="text-sm font-medium group-hover:text-primary">{link.label}</span>
                <ArrowUpRight className="ml-auto size-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}

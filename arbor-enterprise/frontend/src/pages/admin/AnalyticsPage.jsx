import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  Loader2,
  AlertCircle,
  RefreshCw,
  Store,
  Building2,
  MapPin,
  Sparkles,
  PieChart as PieChartIcon,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

import { cn, formatNumber, formatPercentage } from '@/lib/utils';
import { useAdminStore } from '@/stores/adminStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const CHART_COLORS = [
  'hsl(var(--chart-1))',
  'hsl(var(--chart-2))',
  'hsl(var(--chart-3))',
  'hsl(var(--chart-4))',
  'hsl(var(--chart-5))',
];

// Fallback colors if CSS vars not defined
const FALLBACK_COLORS = [
  '#6366f1',
  '#22c55e',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
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

function CustomTooltipContent({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-card p-3 shadow-md">
      <p className="mb-1 text-xs font-medium">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="text-xs" style={{ color: entry.color }}>
          {entry.name}: {entry.value}
        </p>
      ))}
    </div>
  );
}

export default function AdminAnalyticsPage() {
  const {
    stats,
    statsLoading,
    statsError,
    metrics,
    metricsLoading,
    metricsError,
    fetchStats,
    fetchMetrics,
  } = useAdminStore();

  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchStats();
    fetchMetrics();
  }, [fetchStats, fetchMetrics]);

  async function handleRefresh() {
    setRefreshing(true);
    try {
      await Promise.all([fetchStats(), fetchMetrics()]);
    } finally {
      setRefreshing(false);
    }
  }

  // Build pie chart data from stats
  const distributionData = stats
    ? [
        { name: 'Brands', value: stats.total_brands || 0 },
        { name: 'Venues', value: stats.total_venues || 0 },
      ].filter((d) => d.value > 0)
    : [];

  // Build enrichment progress bar chart
  const enrichmentData = stats
    ? [
        {
          name: 'Enriched',
          value: stats.enriched_entities || 0,
        },
        {
          name: 'Not Enriched',
          value: (stats.total_entities || 0) - (stats.enriched_entities || 0),
        },
      ]
    : [];

  const enrichmentRate = stats?.total_entities
    ? ((stats.enriched_entities || 0) / stats.total_entities) * 100
    : 0;

  // Metrics chart data
  const metricsChartData = metrics?.chart_data || metrics?.data || metrics?.timeseries || [];

  const kpis = [
    {
      title: 'Total Entities',
      value: formatNumber(stats?.total_entities ?? 0),
      icon: Store,
      description: 'All entity types combined',
    },
    {
      title: 'Brands',
      value: formatNumber(stats?.total_brands ?? 0),
      icon: Building2,
      description: distributionData.length > 0
        ? formatPercentage((stats?.total_brands / stats?.total_entities) * 100)
        : 'Brand entities',
    },
    {
      title: 'Venues',
      value: formatNumber(stats?.total_venues ?? 0),
      icon: MapPin,
      description: distributionData.length > 0
        ? formatPercentage((stats?.total_venues / stats?.total_entities) * 100)
        : 'Venue entities',
    },
    {
      title: 'Enrichment Rate',
      value: formatPercentage(enrichmentRate),
      icon: Sparkles,
      description: `${formatNumber(stats?.enriched_entities ?? 0)} enriched`,
    },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <BarChart3 className="size-5" />
            Analytics
          </h2>
          <p className="text-sm text-muted-foreground">
            Data insights and enrichment metrics
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

      {/* Error States */}
      {(statsError || metricsError) && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="flex items-center gap-3 p-4">
            <AlertCircle className="size-5 text-destructive" />
            <div>
              <p className="text-sm font-medium">Failed to load data</p>
              <p className="text-xs text-muted-foreground">{statsError || metricsError}</p>
            </div>
            <Button variant="outline" size="sm" className="ml-auto" onClick={handleRefresh}>
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

      {/* Tabbed Charts */}
      <Tabs defaultValue="distribution">
        <TabsList>
          <TabsTrigger value="distribution">
            <PieChartIcon className="mr-1.5 size-4" />
            Distribution
          </TabsTrigger>
          <TabsTrigger value="enrichment">
            <Sparkles className="mr-1.5 size-4" />
            Enrichment
          </TabsTrigger>
          <TabsTrigger value="metrics">
            <TrendingUp className="mr-1.5 size-4" />
            Metrics
          </TabsTrigger>
        </TabsList>

        {/* Distribution Tab */}
        <TabsContent value="distribution">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Entity Distribution by Type</CardTitle>
              <CardDescription>Breakdown of entities across brands and venues</CardDescription>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="mx-auto h-[300px] w-[300px]" />
              ) : distributionData.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <PieChartIcon className="mb-3 size-8 text-muted-foreground/50" />
                  <p className="text-sm text-muted-foreground">No data available</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-6 md:flex-row md:justify-center">
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={distributionData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={4}
                        dataKey="value"
                      >
                        {distributionData.map((_, i) => (
                          <Cell
                            key={i}
                            fill={FALLBACK_COLORS[i % FALLBACK_COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltipContent />} />
                      <Legend
                        formatter={(value) => (
                          <span className="text-xs text-foreground">{value}</span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enrichment Tab */}
        <TabsContent value="enrichment">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Enrichment Progress</CardTitle>
              <CardDescription>
                {enrichmentRate > 0
                  ? `${formatPercentage(enrichmentRate)} of entities have been enriched`
                  : 'Track enrichment completion across all entities'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="h-[300px] w-full" />
              ) : enrichmentData.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <Sparkles className="mb-3 size-8 text-muted-foreground/50" />
                  <p className="text-sm text-muted-foreground">No enrichment data</p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={enrichmentData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      type="number"
                      tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                      axisLine={{ stroke: 'hsl(var(--border))' }}
                    />
                    <YAxis
                      dataKey="name"
                      type="category"
                      tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                      axisLine={{ stroke: 'hsl(var(--border))' }}
                      width={100}
                    />
                    <Tooltip content={<CustomTooltipContent />} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {enrichmentData.map((_, i) => (
                        <Cell
                          key={i}
                          fill={i === 0 ? FALLBACK_COLORS[0] : FALLBACK_COLORS[3]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">System Metrics</CardTitle>
              <CardDescription>Performance and usage metrics over time</CardDescription>
            </CardHeader>
            <CardContent>
              {metricsLoading ? (
                <Skeleton className="h-[300px] w-full" />
              ) : metricsChartData.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <TrendingUp className="mb-3 size-8 text-muted-foreground/50" />
                  <p className="text-sm text-muted-foreground">No metrics data available</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Metrics will appear once the system collects enough data
                  </p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={metricsChartData}>
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
                    <Tooltip content={<CustomTooltipContent />} />
                    <Bar
                      dataKey="value"
                      fill={FALLBACK_COLORS[0]}
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

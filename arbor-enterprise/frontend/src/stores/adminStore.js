import { create } from 'zustand';
import { apiGet, apiPost, apiDelete } from '@/lib/api';

/**
 * Admin store -- manages admin dashboard state including stats, health,
 * entities list, metrics, and enrichment operations.
 */
export const useAdminStore = create((set, get) => ({
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  stats: null,
  statsLoading: false,
  statsError: null,

  health: null,
  healthLoading: false,
  healthError: null,

  metrics: null,
  metricsLoading: false,
  metricsError: null,

  entities: [],
  entitiesTotal: 0,
  entitiesLoading: false,
  entitiesError: null,

  enrichmentStatus: null,
  enrichmentLoading: false,

  loading: false,
  error: null,

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  fetchStats: async () => {
    set({ statsLoading: true, statsError: null });
    try {
      const data = await apiGet('/api/v1/admin/stats');
      set({ stats: data, statsLoading: false });
      return data;
    } catch (err) {
      set({ statsError: err.message, statsLoading: false });
    }
  },

  fetchHealth: async () => {
    set({ healthLoading: true, healthError: null });
    try {
      const data = await apiGet('/api/v1/admin/health/readiness');
      set({ health: data, healthLoading: false });
      return data;
    } catch (err) {
      set({ healthError: err.message, healthLoading: false });
    }
  },

  fetchMetrics: async () => {
    set({ metricsLoading: true, metricsError: null });
    try {
      const data = await apiGet('/api/v1/admin/metrics');
      set({ metrics: data, metricsLoading: false });
      return data;
    } catch (err) {
      set({ metricsError: err.message, metricsLoading: false });
    }
  },

  fetchEntities: async (params = {}) => {
    set({ entitiesLoading: true, entitiesError: null });
    try {
      const data = await apiGet('/api/v1/entities', params);
      set({
        entities: data.items || [],
        entitiesTotal: data.total || 0,
        entitiesLoading: false,
      });
      return data;
    } catch (err) {
      set({ entitiesError: err.message, entitiesLoading: false });
    }
  },

  enrichEntity: async (entityId) => {
    return apiPost(`/api/v1/admin/enrich/${entityId}`);
  },

  cancelEnrichment: async (entityId) => {
    return apiDelete(`/api/v1/admin/enrich/${entityId}`);
  },

  fetchEnrichmentStatus: async () => {
    set({ enrichmentLoading: true });
    try {
      const data = await apiGet('/api/v1/curator/status');
      set({ enrichmentStatus: data, enrichmentLoading: false });
      return data;
    } catch (err) {
      set({ enrichmentLoading: false });
    }
  },

  fetchAll: async () => {
    set({ loading: true, error: null });
    try {
      await Promise.all([
        get().fetchStats(),
        get().fetchHealth(),
        get().fetchMetrics(),
      ]);
    } finally {
      set({ loading: false });
    }
  },
}));

import { create } from 'zustand';
import { apiGet, apiPost, apiDelete } from '@/lib/api';

export const useCuratorStore = create((set, get) => ({
  reviewQueue: [],
  goldStandard: [],
  driftReport: null,
  feedbackAnalysis: null,
  enrichmentStatus: null,
  loading: false,
  error: null,

  fetchReviewQueue: async () => {
    set({ loading: true, error: null });
    try {
      const data = await apiGet('/api/v1/curator/review-queue');
      set({ reviewQueue: Array.isArray(data) ? data : data.items || [], loading: false });
    } catch (err) {
      set({ error: err.message, loading: false });
    }
  },

  decideReview: async (itemId, action, overrideData, notes) => {
    const body = { action };
    if (overrideData) body.override_data = overrideData;
    if (notes) body.notes = notes;

    await apiPost(`/api/v1/curator/review-queue/${itemId}/decide`, body);
    set((state) => ({
      reviewQueue: state.reviewQueue.filter((item) => item.id !== itemId),
    }));
  },

  fetchGoldStandard: async () => {
    set({ loading: true, error: null });
    try {
      const data = await apiGet('/api/v1/curator/gold-standard');
      set({ goldStandard: Array.isArray(data) ? data : data.items || [], loading: false });
    } catch (err) {
      set({ error: err.message, loading: false });
    }
  },

  addGoldStandard: async (entry) => {
    const data = await apiPost('/api/v1/curator/gold-standard', entry);
    set((state) => ({
      goldStandard: [...state.goldStandard, data],
    }));
    return data;
  },

  deleteGoldStandard: async (id) => {
    await apiDelete(`/api/v1/curator/gold-standard/${id}`);
    set((state) => ({
      goldStandard: state.goldStandard.filter((item) => item.id !== id),
    }));
  },

  fetchDriftReport: async () => {
    set({ loading: true, error: null });
    try {
      const data = await apiGet('/api/v1/curator/drift-report');
      set({ driftReport: data, loading: false });
    } catch (err) {
      set({ error: err.message, loading: false });
    }
  },

  fetchFeedbackAnalysis: async () => {
    try {
      const data = await apiGet('/api/v1/curator/feedback-analysis');
      set({ feedbackAnalysis: data });
    } catch (err) {
      set({ error: err.message });
    }
  },

  fetchEnrichmentStatus: async () => {
    try {
      const data = await apiGet('/api/v1/curator/status');
      set({ enrichmentStatus: data });
    } catch (err) {
      set({ error: err.message });
    }
  },
}));

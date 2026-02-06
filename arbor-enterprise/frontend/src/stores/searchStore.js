import { create } from 'zustand';
import { apiGet } from '@/lib/api';

export const useSearchStore = create((set, get) => ({
  results: [],
  loading: false,
  error: null,
  query: '',

  search: async (query, options = {}) => {
    if (!query.trim()) {
      set({ results: [], query: '' });
      return;
    }

    set({ loading: true, error: null, query });

    const params = { query, ...options };
    if (!params.limit) params.limit = 20;

    try {
      const data = await apiGet('/api/v1/search/vector', params);
      set({
        results: Array.isArray(data) ? data : data.results || data.items || [],
        loading: false,
      });
    } catch (err) {
      set({ error: err.message, loading: false });
    }
  },

  clear: () => {
    set({ results: [], query: '', error: null });
  },
}));

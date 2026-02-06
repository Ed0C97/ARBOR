import { create } from 'zustand';
import { apiGet } from '@/lib/api';

const PAGE_SIZE = 24;

export const useEntityStore = create((set, get) => ({
  entities: [],
  total: 0,
  loading: false,
  error: null,
  currentPage: 1,
  hasMore: false,
  filters: {
    search: '',
    type: '',
    category: '',
    city: '',
  },

  setFilter: (key, value) => {
    set((state) => ({
      filters: { ...state.filters, [key]: value },
      currentPage: 1,
    }));
    get().fetchEntities();
  },

  fetchEntities: async () => {
    const { currentPage, filters } = get();
    set({ loading: true, error: null });

    const params = {
      limit: PAGE_SIZE,
      offset: (currentPage - 1) * PAGE_SIZE,
    };

    if (filters.search) params.search = filters.search;
    if (filters.type) params.entity_type = filters.type;
    if (filters.category) params.category = filters.category;
    if (filters.city) params.city = filters.city;

    try {
      const data = await apiGet('/api/v1/entities', params);
      set({
        entities: data.items || [],
        total: data.total || 0,
        hasMore: (data.items?.length || 0) >= PAGE_SIZE,
        loading: false,
      });
    } catch (err) {
      set({ error: err.message, loading: false });
    }
  },

  nextPage: () => {
    set((state) => ({ currentPage: state.currentPage + 1 }));
    get().fetchEntities();
  },

  prevPage: () => {
    set((state) => ({
      currentPage: Math.max(1, state.currentPage - 1),
    }));
    get().fetchEntities();
  },
}));

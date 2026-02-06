import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { apiFetch, registerAuthStore } from '@/lib/api';

/**
 * Auth store -- manages user identity, JWT token, and authentication state.
 *
 * The token is persisted in localStorage so that page reloads do not require
 * a fresh login.  In development mode, `loginDev()` provides a one-click
 * login flow that uses the hard-coded `dev-token` and a synthetic admin user.
 */
export const useAuthStore = create(
  persist(
    (set, get) => ({
      // ------------------------------------------------------------------
      // State
      // ------------------------------------------------------------------
      user: null,     // { id, email, name, role, preferences } | null
      token: null,    // JWT string | null
      isAuthenticated: false,

      // ------------------------------------------------------------------
      // Actions
      // ------------------------------------------------------------------

      /**
       * Authenticate with email + password against the backend.
       * On success the store is populated with the user object and token.
       */
      login: async (email, password) => {
        try {
          const data = await apiFetch('/api/v1/auth/login', {
            method: 'POST',
            body: { email, password },
          });

          set({
            user: data.user,
            token: data.token ?? data.access_token,
            isAuthenticated: true,
          });

          return data;
        } catch (err) {
          set({ user: null, token: null, isAuthenticated: false });
          throw err;
        }
      },

      /**
       * Development-only shortcut.  Sets a hard-coded token (`dev-token`)
       * and a synthetic admin user so the frontend can be exercised without
       * a running auth service.
       */
      loginDev: () => {
        set({
          user: {
            id: 'dev-user-001',
            email: 'admin@arbor.dev',
            name: 'Dev Admin',
            role: 'admin',
            preferences: {
              theme: 'dark',
              defaultCity: null,
              defaultCountry: null,
            },
          },
          token: 'dev-token',
          isAuthenticated: true,
        });
      },

      /**
       * Clear all auth state and remove the persisted token.
       */
      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },

      /**
       * Replace the current user object (e.g. after a profile update).
       */
      setUser: (user) => {
        set({ user });
      },

      /**
       * Directly set the auth token (useful for OAuth / SSO flows).
       */
      setToken: (token) => {
        set({
          token,
          isAuthenticated: !!token,
        });
      },
    }),
    {
      name: 'arbor-auth',           // localStorage key
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);

// ---------------------------------------------------------------------------
// Register with the API layer so it can read the token without a circular
// import at module-evaluation time.
// ---------------------------------------------------------------------------
registerAuthStore(useAuthStore);

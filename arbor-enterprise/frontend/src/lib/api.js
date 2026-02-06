import { toast } from 'sonner';

/**
 * Centralized API client for the ARBOR Enterprise backend.
 *
 * All requests are routed through the Vite dev-server proxy, so only
 * relative paths ("/api/v1/...") are needed.  The auth token is read
 * lazily from the auth store to avoid circular-import issues at module
 * load time.
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Lazily resolved reference to the auth store (avoids circular imports). */
let _authStore = null;

function getAuthStore() {
  if (!_authStore) {
    // Dynamic import would be async; instead we rely on the store module
    // exporting a vanilla store that is available synchronously after first
    // import.  The store file sets this reference via `registerAuthStore`.
    // As a fallback, we attempt a synchronous require-style import.
    try {
      const mod = require('@/stores/authStore');
      _authStore = mod.useAuthStore;
    } catch {
      // Will be set externally via registerAuthStore()
    }
  }
  return _authStore;
}

/**
 * Called by authStore on module init to hand its store reference to the API
 * layer without creating a circular dependency at import time.
 */
export function registerAuthStore(store) {
  _authStore = store;
}

function getToken() {
  const store = getAuthStore();
  if (store) {
    return store.getState().token;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Perform an authenticated fetch against the ARBOR API.
 *
 * @param {string}  path     - API path, e.g. "/api/v1/entities"
 * @param {object}  [options]
 * @param {string}  [options.method]  - HTTP method (default "GET")
 * @param {object}  [options.body]    - Will be JSON-stringified automatically
 * @param {object}  [options.headers] - Additional headers (merged with defaults)
 * @param {object}  [options.params]  - URL search params (appended to path)
 * @param {boolean} [options.raw]     - If true, return the raw Response instead of parsed JSON
 * @param {AbortSignal} [options.signal] - AbortController signal
 * @returns {Promise<any>} Parsed JSON response body
 * @throws {ApiError} On non-2xx responses
 */
export async function apiFetch(path, options = {}) {
  const {
    method = 'GET',
    body,
    headers: extraHeaders = {},
    params,
    raw = false,
    signal,
  } = options;

  // Build URL with optional query params
  let url = path;
  if (params) {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, String(value));
      }
    }
    const qs = searchParams.toString();
    if (qs) {
      url += (url.includes('?') ? '&' : '?') + qs;
    }
  }

  // Headers
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...extraHeaders,
  };

  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Remove Content-Type for requests without a body (GET, DELETE without body)
  if (!body) {
    delete headers['Content-Type'];
  }

  const fetchOptions = {
    method,
    headers,
    signal,
  };

  if (body) {
    fetchOptions.body = JSON.stringify(body);
  }

  // -----------------------------------------------------------------------
  // Execute
  // -----------------------------------------------------------------------
  let response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (err) {
    if (err.name === 'AbortError') {
      throw err;
    }
    toast.error('Network error. Please check your connection.');
    throw new ApiError('Network error', 0, null);
  }

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------
  if (!response.ok) {
    let errorBody = null;
    try {
      errorBody = await response.json();
    } catch {
      // Non-JSON error body -- ignore
    }

    const message =
      errorBody?.detail ||
      errorBody?.message ||
      `Request failed with status ${response.status}`;

    switch (response.status) {
      case 401: {
        // Unauthorized -- clear auth state and redirect to login
        const store = getAuthStore();
        if (store) {
          store.getState().logout();
        }
        // Only redirect if we're not already on the login page
        if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
          window.location.href = '/login';
        }
        break;
      }
      case 429:
        toast.error('Rate limit exceeded. Please slow down and try again.');
        break;
      case 500:
      case 502:
      case 503:
        toast.error(`Server error: ${message}`);
        break;
      default:
        // 4xx client errors -- surface but don't toast automatically
        break;
    }

    throw new ApiError(message, response.status, errorBody);
  }

  // -----------------------------------------------------------------------
  // Success
  // -----------------------------------------------------------------------
  if (raw) {
    return response;
  }

  // 204 No Content
  if (response.status === 204) {
    return null;
  }

  // Parse JSON
  try {
    return await response.json();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Error class
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status   - HTTP status code (0 for network errors)
   * @param {any}    body     - Parsed response body, if available
   */
  constructor(message, status, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

// ---------------------------------------------------------------------------
// Convenience methods
// ---------------------------------------------------------------------------

export function apiGet(path, params, options = {}) {
  return apiFetch(path, { method: 'GET', params, ...options });
}

export function apiPost(path, body, options = {}) {
  return apiFetch(path, { method: 'POST', body, ...options });
}

export function apiPut(path, body, options = {}) {
  return apiFetch(path, { method: 'PUT', body, ...options });
}

export function apiPatch(path, body, options = {}) {
  return apiFetch(path, { method: 'PATCH', body, ...options });
}

export function apiDelete(path, options = {}) {
  return apiFetch(path, { method: 'DELETE', ...options });
}

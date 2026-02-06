/**
 * SSE (Server-Sent Events) client for the ARBOR streaming discovery endpoint.
 *
 * Uses fetch + ReadableStream instead of the native EventSource API because
 * the endpoint requires a POST body with query and filter parameters.
 *
 * Wire format from the server:
 *   event: <type>\n
 *   data: <json>\n
 *   \n
 *
 * Supported event types:
 *   status          - Progress updates (e.g. "Searching...", "Analyzing...")
 *   recommendation  - A single entity recommendation object
 *   text            - Incremental text chunk (streamed prose)
 *   error           - Server-side error
 *   done            - Stream complete
 */

import { useAuthStore } from '@/stores/authStore';

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

/**
 * Parses a raw SSE text block into discrete events.
 *
 * @param {string} raw - The raw text that may contain one or more SSE events
 * @returns {{ type: string, data: any }[]}
 */
function parseSSEChunk(raw) {
  const events = [];
  // Split on double-newline boundaries (event separator)
  const blocks = raw.split(/\n\n/);

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    let eventType = 'message';
    let dataLines = [];

    for (const line of trimmed.split('\n')) {
      if (line.startsWith('event:')) {
        eventType = line.slice('event:'.length).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).trim());
      }
      // Ignore id:, retry:, and comment lines (starting with ':')
    }

    if (dataLines.length === 0) continue;

    const dataStr = dataLines.join('\n');
    let data;
    try {
      data = JSON.parse(dataStr);
    } catch {
      // Non-JSON data -- keep as plain string
      data = dataStr;
    }

    events.push({ type: eventType, data });
  }

  return events;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create a streaming discovery request.
 *
 * @param {string} query    - The natural-language discovery query
 * @param {object} [options]
 * @param {string} [options.city]     - City filter
 * @param {string} [options.country]  - Country filter
 * @param {string} [options.type]     - Entity type filter (brand | venue)
 * @param {string} [options.category] - Category filter
 * @param {number} [options.limit]    - Max recommendations
 * @param {object} callbacks
 * @param {(status: string) => void}           [callbacks.onStatus]
 * @param {(recommendation: object) => void}   [callbacks.onRecommendation]
 * @param {(text: string) => void}             [callbacks.onText]
 * @param {(error: { message: string }) => void} [callbacks.onError]
 * @param {(summary: object) => void}          [callbacks.onDone]
 * @returns {{ abort: () => void }} Controller with an abort method
 */
export function createDiscoverStream(query, options = {}, callbacks = {}) {
  const {
    onStatus = () => {},
    onRecommendation = () => {},
    onText = () => {},
    onError = () => {},
    onDone = () => {},
  } = callbacks;

  const controller = new AbortController();

  // Build request body
  const body = {
    query,
    ...options,
  };

  // Auth token
  const token = useAuthStore.getState().token;
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Fire-and-forget the async work
  (async () => {
    let response;
    try {
      response = await fetch('/api/v1/discover/stream', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      onError({ message: 'Network error: unable to reach the server.' });
      return;
    }

    if (!response.ok) {
      let detail = `Server responded with ${response.status}`;
      try {
        const errBody = await response.json();
        detail = errBody.detail || errBody.message || detail;
      } catch {
        // ignore
      }
      onError({ message: detail });
      return;
    }

    // ---------------------------------------------------------------------------
    // Read the stream
    // ---------------------------------------------------------------------------
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process all complete events in the buffer.
        // A complete event ends with \n\n.  We keep any trailing incomplete
        // fragment in the buffer for the next iteration.
        const lastDoubleNewline = buffer.lastIndexOf('\n\n');
        if (lastDoubleNewline === -1) continue;

        const complete = buffer.slice(0, lastDoubleNewline + 2);
        buffer = buffer.slice(lastDoubleNewline + 2);

        const events = parseSSEChunk(complete);

        for (const event of events) {
          switch (event.type) {
            case 'status':
              onStatus(event.data);
              break;
            case 'recommendation':
              onRecommendation(event.data);
              break;
            case 'text':
              onText(event.data);
              break;
            case 'error':
              onError(typeof event.data === 'string' ? { message: event.data } : event.data);
              break;
            case 'done':
              onDone(event.data);
              break;
            default:
              // Unknown event type -- ignore
              break;
          }
        }
      }

      // Process any remaining data in the buffer
      if (buffer.trim()) {
        const events = parseSSEChunk(buffer);
        for (const event of events) {
          switch (event.type) {
            case 'status':
              onStatus(event.data);
              break;
            case 'recommendation':
              onRecommendation(event.data);
              break;
            case 'text':
              onText(event.data);
              break;
            case 'error':
              onError(typeof event.data === 'string' ? { message: event.data } : event.data);
              break;
            case 'done':
              onDone(event.data);
              break;
            default:
              break;
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      onError({ message: err.message || 'Stream reading failed.' });
    }
  })();

  return {
    abort: () => controller.abort(),
  };
}

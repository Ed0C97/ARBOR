import { create } from 'zustand';
import { createDiscoverStream } from '@/lib/sse';

/**
 * Chat / Discover store -- manages the conversational discovery UI.
 *
 * Each user query triggers a streaming request.  Intermediate SSE events
 * accumulate text chunks and recommendation objects, and a final assistant
 * message is materialised when the stream completes.
 */

let _currentStream = null; // reference to the active stream's abort handle

export const useChatStore = create((set, get) => ({
  // --------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------

  /** @type {{ id: string, role: 'user'|'assistant', content: string, recommendations: object[], timestamp: number }[]} */
  messages: [],

  /** True while a stream is in progress. */
  isStreaming: false,

  /** Accumulated text from `text` SSE events for the current stream. */
  streamingText: '',

  /** Accumulated recommendations from the current stream. */
  streamingRecommendations: [],

  /** Latest status label from the stream (e.g. "Searching...", "Ranking..."). */
  currentStatus: null,

  /** Last error message, if any. */
  error: null,

  // --------------------------------------------------------------------
  // Actions
  // --------------------------------------------------------------------

  /**
   * Send a discovery query and begin streaming the response.
   *
   * @param {string} query    - Natural language query
   * @param {object} [filters] - Optional filters (city, country, type, category, limit)
   */
  sendQuery: (query, filters = {}) => {
    const state = get();

    // If already streaming, cancel the previous one first
    if (state.isStreaming && _currentStream) {
      _currentStream.abort();
    }

    // Add the user message immediately
    const userMessage = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: query,
      recommendations: [],
      timestamp: Date.now(),
    };

    set({
      messages: [...get().messages, userMessage],
      isStreaming: true,
      streamingText: '',
      streamingRecommendations: [],
      currentStatus: 'Connecting...',
      error: null,
    });

    _currentStream = createDiscoverStream(query, filters, {
      onStatus: (data) => {
        const status = typeof data === 'string' ? data : data.status || data.message || 'Processing...';
        set({ currentStatus: status });
      },

      onRecommendation: (recommendation) => {
        set({
          streamingRecommendations: [...get().streamingRecommendations, recommendation],
        });
      },

      onText: (data) => {
        const chunk = typeof data === 'string' ? data : data.text || data.content || '';
        set({
          streamingText: get().streamingText + chunk,
        });
      },

      onError: (err) => {
        const message = err?.message || 'An error occurred during discovery.';
        set({
          error: message,
          isStreaming: false,
          currentStatus: null,
        });

        // Still materialise whatever we collected so far
        const { streamingText, streamingRecommendations } = get();
        if (streamingText || streamingRecommendations.length > 0) {
          const assistantMessage = {
            id: `msg-${Date.now()}-assistant`,
            role: 'assistant',
            content: streamingText || message,
            recommendations: streamingRecommendations,
            timestamp: Date.now(),
          };
          set({
            messages: [...get().messages, assistantMessage],
            streamingText: '',
            streamingRecommendations: [],
          });
        }

        _currentStream = null;
      },

      onDone: (summary) => {
        const { streamingText, streamingRecommendations } = get();

        const assistantMessage = {
          id: `msg-${Date.now()}-assistant`,
          role: 'assistant',
          content: streamingText,
          recommendations: streamingRecommendations,
          timestamp: Date.now(),
        };

        set({
          messages: [...get().messages, assistantMessage],
          isStreaming: false,
          streamingText: '',
          streamingRecommendations: [],
          currentStatus: null,
        });

        _currentStream = null;
      },
    });
  },

  /**
   * Cancel the current streaming request, if any.
   */
  cancelStream: () => {
    if (_currentStream) {
      _currentStream.abort();
      _currentStream = null;
    }

    const { streamingText, streamingRecommendations } = get();

    // Materialise whatever was collected before cancellation
    if (streamingText || streamingRecommendations.length > 0) {
      const assistantMessage = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: streamingText || '(Cancelled)',
        recommendations: streamingRecommendations,
        timestamp: Date.now(),
      };
      set({
        messages: [...get().messages, assistantMessage],
      });
    }

    set({
      isStreaming: false,
      streamingText: '',
      streamingRecommendations: [],
      currentStatus: null,
      error: null,
    });
  },

  /**
   * Reset the entire chat history.
   */
  clearChat: () => {
    if (_currentStream) {
      _currentStream.abort();
      _currentStream = null;
    }

    set({
      messages: [],
      isStreaming: false,
      streamingText: '',
      streamingRecommendations: [],
      currentStatus: null,
      error: null,
    });
  },

  /**
   * Manually add a message (useful for system messages or restoring history).
   */
  addMessage: (msg) => {
    set({
      messages: [
        ...get().messages,
        {
          id: msg.id || `msg-${Date.now()}`,
          role: msg.role || 'assistant',
          content: msg.content || '',
          recommendations: msg.recommendations || [],
          timestamp: msg.timestamp || Date.now(),
        },
      ],
    });
  },
}));

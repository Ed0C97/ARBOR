import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  StopCircle,
  Sparkles,
  MapPin,
  Star,
  ExternalLink,
  Loader2,
  Trees,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { useChatStore } from '@/stores/chatStore';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';

function RecommendationCard({ entity }) {
  return (
    <Link to={`/entity/${entity.id}`}>
      <Card className="group cursor-pointer p-3 transition-colors hover:bg-accent/50">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <h4 className="truncate text-sm font-semibold group-hover:text-primary">
              {entity.name}
            </h4>
            <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
              {entity.category && <Badge variant="secondary" className="text-[10px] px-1.5 py-0">{entity.category}</Badge>}
              {entity.city && (
                <span className="flex items-center gap-0.5">
                  <MapPin className="size-3" />
                  {entity.city}
                </span>
              )}
            </div>
            {entity.description && (
              <p className="mt-1.5 line-clamp-2 text-xs text-muted-foreground">
                {entity.description}
              </p>
            )}
          </div>
          <div className="flex flex-col items-end gap-1">
            {entity.rating && (
              <span className="flex items-center gap-0.5 text-xs font-medium text-amber-500">
                <Star className="size-3 fill-current" />
                {entity.rating}
              </span>
            )}
            {entity.price_range && (
              <span className="text-xs text-muted-foreground">
                {entity.price_range}
              </span>
            )}
            <ExternalLink className="size-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
          </div>
        </div>
      </Card>
    </Link>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'flex gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row',
      )}
    >
      <div
        className={cn(
          'flex size-8 shrink-0 items-center justify-center rounded-lg text-xs font-medium',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground',
        )}
      >
        {isUser ? 'You' : <Trees className="size-4" />}
      </div>

      <div
        className={cn(
          'max-w-[75%] space-y-3',
          isUser ? 'text-right' : 'text-left',
        )}
      >
        <div
          className={cn(
            'inline-block rounded-xl px-4 py-2.5 text-sm leading-relaxed',
            isUser
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted/60',
          )}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {message.recommendations?.length > 0 && (
          <div className="space-y-2">
            {message.recommendations.map((entity, i) => (
              <RecommendationCard key={entity.id || i} entity={entity} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function StreamingIndicator({ status, text, recommendations }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3"
    >
      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
        <Trees className="size-4" />
      </div>

      <div className="max-w-[75%] space-y-3">
        {status && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            <span>{status}</span>
          </div>
        )}

        {text && (
          <div className="inline-block rounded-xl bg-muted/60 px-4 py-2.5 text-sm leading-relaxed">
            <p className="whitespace-pre-wrap">{text}</p>
            <span className="inline-block h-4 w-0.5 animate-pulse bg-foreground" />
          </div>
        )}

        {recommendations?.length > 0 && (
          <div className="space-y-2">
            {recommendations.map((entity, i) => (
              <RecommendationCard key={entity.id || i} entity={entity} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

const SUGGESTIONS = [
  'Find the best tailors in London',
  'Recommend vintage menswear shops in Milan',
  'Where can I find artisan leather goods in Florence?',
  'Show me top-rated streetwear brands in Tokyo',
];

export default function DiscoverPage() {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const {
    messages,
    isStreaming,
    streamingText,
    streamingRecommendations,
    currentStatus,
    sendQuery,
    cancelStream,
  } = useChatStore();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText, streamingRecommendations]);

  function handleSubmit(e) {
    e.preventDefault();
    const query = input.trim();
    if (!query || isStreaming) return;
    setInput('');
    sendQuery(query);
  }

  function handleSuggestion(text) {
    setInput('');
    sendQuery(text);
  }

  const isEmpty = messages.length === 0 && !isStreaming;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Messages area */}
      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-4 py-6">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <Sparkles className="size-8" />
              </div>
              <h2 className="mb-2 text-xl font-semibold">
                What would you like to discover?
              </h2>
              <p className="mb-8 text-center text-sm text-muted-foreground max-w-md">
                Ask me anything about fashion brands, boutiques, tailors,
                artisans and venues around the world.
              </p>

              <div className="grid w-full max-w-lg gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestion(suggestion)}
                    className="rounded-lg border bg-card p-3 text-left text-sm transition-colors hover:bg-accent"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <AnimatePresence mode="popLayout">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))}
              </AnimatePresence>

              {isStreaming && (
                <StreamingIndicator
                  status={currentStatus}
                  text={streamingText}
                  recommendations={streamingRecommendations}
                />
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input area */}
      <div className="border-t bg-background/80 backdrop-blur-sm">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-2 p-4"
        >
          <div className="relative flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask ARBOR anything..."
              rows={1}
              className="w-full resize-none rounded-xl border bg-background px-4 py-3 pr-12 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>

          {isStreaming ? (
            <Button
              type="button"
              size="icon"
              variant="destructive"
              className="size-10 shrink-0 rounded-xl"
              onClick={cancelStream}
            >
              <StopCircle className="size-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              className="size-10 shrink-0 rounded-xl"
              disabled={!input.trim()}
            >
              <Send className="size-4" />
            </Button>
          )}
        </form>
      </div>
    </div>
  );
}

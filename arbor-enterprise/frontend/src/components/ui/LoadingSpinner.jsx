import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

const sizes = {
  sm: 'size-4',
  md: 'size-6',
  lg: 'size-8',
  xl: 'size-12',
};

export default function LoadingSpinner({ size = 'md', className }) {
  return (
    <Loader2
      className={cn('animate-spin text-muted-foreground', sizes[size], className)}
    />
  );
}

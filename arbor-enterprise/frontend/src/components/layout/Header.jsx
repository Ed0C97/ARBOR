import { useLocation } from 'react-router-dom';
import { Sun, Moon, Search } from 'lucide-react';

import { useTheme } from '@/components/ThemeProvider';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from '@/components/ui/tooltip';

const pageTitles = {
  '/': 'Discover',
  '/browse': 'Browse',
  '/entities': 'Entities',
  '/graph': 'Knowledge Graph',
  '/profile': 'Profile',
  '/settings': 'Settings',
  '/login': 'Login',
  '/curator/review': 'Review Queue',
  '/curator/gold-standard': 'Gold Standard',
  '/curator/drift': 'Drift Report',
  '/admin': 'Admin Dashboard',
  '/admin/entities': 'Entity Management',
  '/admin/ingestion': 'Data Ingestion',
  '/admin/analytics': 'Analytics',
  '/admin/health': 'System Health',
};

function getPageTitle(pathname) {
  if (pageTitles[pathname]) return pageTitles[pathname];
  if (pathname.startsWith('/entity/')) return 'Entity Detail';
  return 'ARBOR';
}

export function Header() {
  const { theme, setTheme } = useTheme();
  const location = useLocation();
  const title = getPageTitle(location.pathname);

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-2 h-4" />

      <h1 className="text-sm font-semibold tracking-tight">{title}</h1>

      <div className="ml-auto flex items-center gap-1">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {theme === 'dark' ? (
                  <Sun className="size-4" />
                ) : (
                  <Moon className="size-4" />
                )}
                <span className="sr-only">Toggle theme</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Switch to {theme === 'dark' ? 'light' : 'dark'} mode
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </header>
  );
}

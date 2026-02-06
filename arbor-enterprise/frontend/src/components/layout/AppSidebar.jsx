import { useLocation, Link } from 'react-router-dom';
import {
  MessageSquare,
  Compass,
  Store,
  Network,
  User,
  LayoutDashboard,
  Database,
  Download,
  BarChart3,
  Activity,
  ClipboardCheck,
  Award,
  TrendingDown,
  Settings,
  LogOut,
  Trees,
  ChevronDown,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/authStore';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
} from '@/components/ui/sidebar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';

const userNav = [
  { title: 'Discover', icon: MessageSquare, href: '/' },
  { title: 'Browse', icon: Compass, href: '/browse' },
  { title: 'Entities', icon: Store, href: '/entities' },
  { title: 'Graph', icon: Network, href: '/graph' },
];

const curatorNav = [
  { title: 'Review Queue', icon: ClipboardCheck, href: '/curator/review' },
  { title: 'Gold Standard', icon: Award, href: '/curator/gold-standard' },
  { title: 'Drift Report', icon: TrendingDown, href: '/curator/drift' },
];

const adminNav = [
  { title: 'Dashboard', icon: LayoutDashboard, href: '/admin' },
  { title: 'Entities', icon: Database, href: '/admin/entities' },
  { title: 'Ingestion', icon: Download, href: '/admin/ingestion' },
  { title: 'Analytics', icon: BarChart3, href: '/admin/analytics' },
  { title: 'Health', icon: Activity, href: '/admin/health' },
];

function NavItem({ item }) {
  const location = useLocation();
  const isActive =
    item.href === '/'
      ? location.pathname === '/'
      : location.pathname.startsWith(item.href);

  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={isActive} tooltip={item.title}>
        <Link to={item.href}>
          <item.icon />
          <span>{item.title}</span>
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

export function AppSidebar(props) {
  const { user, logout } = useAuthStore();
  const role = user?.role || 'viewer';
  const showCurator = role === 'curator' || role === 'admin';
  const showAdmin = role === 'admin';

  const initials = user?.name
    ? user.name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : 'U';

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link to="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Trees className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">ARBOR</span>
                  <span className="truncate text-xs text-muted-foreground">
                    Enterprise Discovery
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        {/* User navigation */}
        <SidebarGroup>
          <SidebarGroupLabel>Explore</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {userNav.map((item) => (
                <NavItem key={item.href} item={item} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Curator navigation */}
        {showCurator && (
          <>
            <SidebarSeparator />
            <SidebarGroup>
              <SidebarGroupLabel>Curator</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {curatorNav.map((item) => (
                    <NavItem key={item.href} item={item} />
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </>
        )}

        {/* Admin navigation */}
        {showAdmin && (
          <>
            <SidebarSeparator />
            <SidebarGroup>
              <SidebarGroupLabel>Admin</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {adminNav.map((item) => (
                    <NavItem key={item.href} item={item} />
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </>
        )}
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <Avatar className="h-8 w-8 rounded-lg">
                    <AvatarFallback className="rounded-lg text-xs">
                      {initials}
                    </AvatarFallback>
                  </Avatar>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold">
                      {user?.name || 'User'}
                    </span>
                    <span className="truncate text-xs text-muted-foreground">
                      {user?.email || ''}
                    </span>
                  </div>
                  <ChevronDown className="ml-auto size-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
                side="top"
                align="end"
                sideOffset={4}
              >
                <DropdownMenuItem asChild>
                  <Link to="/profile">
                    <User className="mr-2 h-4 w-4" />
                    Profile
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link to="/settings">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={logout}>
                  <LogOut className="mr-2 h-4 w-4" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}

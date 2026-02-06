import React, { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@/components/ThemeProvider';
import { AppLayout } from '@/components/layout/AppLayout';
import AuthGuard from '@/components/auth/AuthGuard';
import RoleGuard from '@/components/auth/RoleGuard';
import { Toaster } from '@/components/ui/sonner';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

// Eager load the main page
import DiscoverPage from '@/pages/DiscoverPage';

// Lazy load all other pages
const BrowsePage = lazy(() => import('@/pages/BrowsePage'));
const EntityDetailPage = lazy(() => import('@/pages/EntityDetailPage'));
const GraphPage = lazy(() => import('@/pages/GraphPage'));
const ProfilePage = lazy(() => import('@/pages/ProfilePage'));
const LoginPage = lazy(() => import('@/pages/LoginPage'));
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'));

// Curator pages
const ReviewQueuePage = lazy(() => import('@/pages/curator/ReviewQueuePage'));
const GoldStandardPage = lazy(() => import('@/pages/curator/GoldStandardPage'));
const DriftReportPage = lazy(() => import('@/pages/curator/DriftReportPage'));

// Admin pages
const AdminDashboardPage = lazy(() => import('@/pages/admin/DashboardPage'));
const AdminEntitiesPage = lazy(() => import('@/pages/admin/EntitiesPage'));
const AdminIngestionPage = lazy(() => import('@/pages/admin/IngestionPage'));
const AdminAnalyticsPage = lazy(() => import('@/pages/admin/AnalyticsPage'));
const AdminHealthPage = lazy(() => import('@/pages/admin/HealthPage'));

const PageLoader = () => (
  <div className="min-h-screen flex items-center justify-center">
    <LoadingSpinner size="lg" />
  </div>
);

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="arbor-theme">
      <Suspense fallback={<PageLoader />}>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<LoginPage />} />

          {/* Authenticated */}
          <Route element={<AuthGuard><AppLayout /></AuthGuard>}>
            <Route path="/" element={<DiscoverPage />} />
            <Route path="/browse" element={<BrowsePage />} />
            <Route path="/entities" element={<BrowsePage />} />
            <Route path="/entity/:id" element={<EntityDetailPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/profile" element={<ProfilePage />} />

            {/* Curator */}
            <Route element={<RoleGuard roles={['curator', 'admin']} />}>
              <Route path="/curator/review" element={<ReviewQueuePage />} />
              <Route path="/curator/gold-standard" element={<GoldStandardPage />} />
              <Route path="/curator/drift" element={<DriftReportPage />} />
            </Route>

            {/* Admin */}
            <Route element={<RoleGuard roles={['admin']} />}>
              <Route path="/admin" element={<AdminDashboardPage />} />
              <Route path="/admin/entities" element={<AdminEntitiesPage />} />
              <Route path="/admin/ingestion" element={<AdminIngestionPage />} />
              <Route path="/admin/analytics" element={<AdminAnalyticsPage />} />
              <Route path="/admin/health" element={<AdminHealthPage />} />
            </Route>

            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </Suspense>

      <Toaster
        position="top-right"
        duration={3000}
        closeButton
        richColors
      />
    </ThemeProvider>
  );
}

export default App;

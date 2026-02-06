import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

export default function RoleGuard({ roles = [] }) {
  const { user } = useAuthStore();
  const userRole = user?.role || 'viewer';

  if (!roles.includes(userRole)) {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
}

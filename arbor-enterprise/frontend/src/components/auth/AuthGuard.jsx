import React, { useEffect } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

export default function AuthGuard({ children }) {
  const { isAuthenticated, loginDev } = useAuthStore();
  const location = useLocation();

  // In development, auto-login with dev token
  useEffect(() => {
    if (!isAuthenticated && import.meta.env.DEV) {
      loginDev();
    }
  }, [isAuthenticated, loginDev]);

  if (!isAuthenticated && !import.meta.env.DEV) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children || <Outlet />;
}

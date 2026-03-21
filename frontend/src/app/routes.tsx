import { createBrowserRouter } from 'react-router';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import { PROTECTED_ROUTES_CHILDREN } from './constants/appRoutes';

export const router = createBrowserRouter([
  {
    path: '/login',
    Component: LoginPage,
  },
  {
    path: '/register',
    Component: RegisterPage,
  },

  {
    Component: ProtectedRoute,
    children: PROTECTED_ROUTES_CHILDREN,
  },
]);

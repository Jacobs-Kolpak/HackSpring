import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from 'react';
import { API_ROUTES } from '../constants/api';
import axiosInstance from '../utils/axiosInstance';

interface User {
  id: number;
  email: string;
  username: string;
  role: 'student' | 'researcher' | 'government';
  is_active: boolean;
  created_at: string;
  updated_at?: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string, role: User['role']) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const refreshAccessToken = useCallback(async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      console.log('refreshAccessToken: No refresh token found.');
      return null;
    }

    try {
      const response = await axiosInstance.post(API_ROUTES.AUTH.REFRESH, {}, {
        headers: {
          Authorization: `Bearer ${refreshToken}`,
        },
      });
      const { access_token, refresh_token } = response.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', refresh_token);
      return access_token;
    } catch (error) {
      console.error('refreshAccessToken: Failed to refresh access token, logging out:', error);
      logout();
      return null;
    }
  }, []);

  const loadUser = useCallback(async () => {
      setLoading(true);
      const accessToken = localStorage.getItem('access_token');
  
      if (!accessToken) {
        console.log('loadUser: No access token found, setting user to null.');
        setUser(null);
        setLoading(false);
        return;
      }
  
      try {
        const response = await axiosInstance.get(API_ROUTES.AUTH.ME);
        setUser(response.data.user);
      } catch (error) {
        console.error('loadUser: Failed to fetch user data with existing access token:', error);
        const newAccessToken = await refreshAccessToken();
  
        if (newAccessToken) {
          try {
            const response = await axiosInstance.get(API_ROUTES.AUTH.ME, {
              headers: {
                Authorization: `Bearer ${newAccessToken}`,
              },
            });
            setUser(response.data.user);
          } catch (refreshError) {
            console.error('loadUser: Failed to fetch user data after refresh, logging out.', refreshError);
            logout();
          }
        } else {
          console.log('loadUser: No new access token after refresh attempt, logging out.');
          logout();
        }
      } finally {
        setLoading(false);
        console.log('loadUser: Loading finished.');
      }
    }, [refreshAccessToken]);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  const login = async (email: string, password: string) => {
    try {
      const response = await axiosInstance.post(API_ROUTES.AUTH.LOGIN, { email, password });
      const { access_token, refresh_token } = response.data;
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', refresh_token);
      await loadUser();
    } catch (error: any) {
        if (error.isAxiosError && error.response) {
        console.error('login: Login failed:', error.response.data.detail);
        throw new Error(error.response.data.detail || 'Login failed');
      }
      console.error('login: Login failed:', error);
      throw new Error('Login failed');
    }
  };

  const register = async (email: string, username: string, password: string, role: User['role']) => {
    try {
      await axiosInstance.post(API_ROUTES.AUTH.REGISTER, { email, username, password, role });
      await login(email, password);
    } catch (error: any) {
      if (error.isAxiosError && error.response) {
        throw new Error(error.response.data.detail || 'Registration failed');
      }
      console.error('register: Registration failed:', error);
      throw new Error('Registration failed');
    }
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    axiosInstance.post(API_ROUTES.AUTH.LOGOUT).catch(console.error);
  };

  return (
    <AuthContext.Provider value={{
      user,
      login,
      register,
      logout,
      isAuthenticated: !!user,
      loading,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { UserPlus, Mail, Lock, User, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import PageClearButton from '../components/PageClearButton';
import {
  DEFAULT_USER_ROLE,
  type UserRole,
  USER_ROLES,
} from '../constants/userRoles';

export default function RegisterPage() {
  const initialFormState = {
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: DEFAULT_USER_ROLE as UserRole,
  };
  const [formData, setFormData] = useState(initialFormState);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const hasClearableOutput =
    Boolean(formData.name) ||
    Boolean(formData.email) ||
    Boolean(formData.password) ||
    Boolean(formData.confirmPassword) ||
    formData.role !== DEFAULT_USER_ROLE ||
    Boolean(error);

  const handleClearPageOutput = () => {
    setFormData(initialFormState);
    setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Пароли не совпадают');
      return;
    }

    if (formData.password.length < 6) {
      setError('Пароль должен быть не менее 6 символов');
      return;
    }

    setIsLoading(true);

    try {
      await register(formData.email, formData.name, formData.password, formData.role);
      navigate('/');
    } catch (err) {
      setError('Ошибка регистрации. Попробуйте еще раз.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || isLoading}
        title="Очистить форму"
      />
      {/* Background gradient effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-[#38C571]/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-[#70D116]/10 rounded-full blur-3xl"></div>
      </div>

      <div className="w-full max-w-md relative z-10">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-3 tracking-tight">Регистрация</h1>
          <p className="text-gray-400 text-lg">Создайте аккаунт в KolpakBook</p>
        </div>

        {/* Register Form */}
        <div className="bg-[#151515] border border-[#262626] rounded p-8 shadow-2xl">
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Name */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Имя пользователя
              </label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  placeholder="Ваше имя"
                  className="w-full pl-12 pr-4 py-3 bg-[#0a0a0a] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#38C571] transition-colors"
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  required
                  placeholder="your@email.com"
                  className="w-full pl-12 pr-4 py-3 bg-[#0a0a0a] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#38C571] transition-colors"
                />
              </div>
            </div>

            {/* Role */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Роль
              </label>
              <div className="grid gap-3">
                {USER_ROLES.map((role) => {
                  const RoleIcon = role.icon;
                  return (
                    <button
                      key={role.value}
                      type="button"
                      onClick={() =>
                        setFormData({ ...formData, role: role.value })
                      }
                      className={`p-4 rounded border-2 transition-all flex items-center gap-4 ${
                        formData.role === role.value
                          ? 'border-[#38C571] bg-[#38C571]/10 shadow-lg shadow-[#38C571]/20'
                          : 'border-[#262626] hover:border-[#404040]'
                      }`}
                    >
                      <div className={`w-12 h-12 rounded bg-gradient-to-tr ${role.color} flex items-center justify-center flex-shrink-0`}>
                        <RoleIcon className="w-6 h-6 text-white" />
                      </div>
                      <div className="text-left flex-1">
                        <div className="font-medium text-white mb-1">{role.label}</div>
                        <div className="text-xs text-gray-400">{role.description}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Пароль
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  required
                  placeholder="••••••••"
                  className="w-full pl-12 pr-4 py-3 bg-[#0a0a0a] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#38C571] transition-colors"
                />
              </div>
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Подтвердите пароль
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="password"
                  value={formData.confirmPassword}
                  onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                  required
                  placeholder="••••••••"
                  className="w-full pl-12 pr-4 py-3 bg-[#0a0a0a] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#38C571] transition-colors"
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full px-6 py-3 bg-gradient-to-tr from-[#38C571] to-[#70D116] text-[#0a0a0a] rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-[#0a0a0a] border-t-transparent rounded-full animate-spin"></div>
                  Регистрация...
                </>
              ) : (
                <>
                  <UserPlus className="w-5 h-5" />
                  Зарегистрироваться
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-[#262626]"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-[#151515] text-gray-500">или</span>
            </div>
          </div>

          {/* Login Link */}
          <div className="text-center">
            <p className="text-gray-400">
              Уже есть аккаунт?{' '}
              <Link
                to="/login"
                className="text-[#38C571] hover:text-[#70D116] transition-colors font-medium"
              >
                Войти
              </Link>
            </p>
          </div>
        </div>

        {/* Info */}
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Регистрируясь, вы соглашаетесь с условиями использования и политикой конфиденциальности
          </p>
        </div>
      </div>
    </div>
  );
}

import { useEffect } from 'react';
import { User, Mail, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router';
import { USER_ROLES } from '../constants/userRoles';

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (!user) {
    return null;
  }

  const currentRole = USER_ROLES.find((role) => role.value === user.role);

  return (
    <div className="min-h-screen p-6 lg:p-12">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Профиль пользователя
          </h1>
          <p className="text-lg text-gray-400">
            Управляйте вашей учетной записью и настройками
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Profile Card */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-[#151515] border border-[#262626] rounded p-6 text-center">
              {/* Avatar */}
              <div className={`w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-tr ${currentRole?.color} flex items-center justify-center shadow-lg`}
                   style={{
                     boxShadow: `0 10px 40px ${currentRole?.profileShadow}`
                   }}>
                <User className="w-12 h-12 text-white" />
              </div>

              <h2 className="text-xl font-bold text-white mb-1">{user.username}</h2>
              <p className="text-sm text-gray-400 mb-6">{user.email}</p>

              {/* Role Card - Prominent */}
              <div className={`p-6 bg-gradient-to-tr ${currentRole?.color} rounded mb-6 shadow-xl`}
                   style={{
                     boxShadow: `0 10px 40px ${currentRole?.profileSoftShadow}`
                   }}>
                {currentRole?.icon && <currentRole.icon className="w-8 h-8 text-white mx-auto mb-3" />}
                <h3 className="text-lg font-bold text-white mb-2">
                  {currentRole?.label}
                </h3>
                {currentRole?.description && (
                  <p className="text-sm text-white/80">
                    {currentRole?.description}
                  </p>
                )}
              </div>

              {/* Logout Button */}
              <button
                onClick={handleLogout}
                className="w-full px-4 py-2 bg-[#0a0a0a] border border-red-600/30 text-red-500 rounded-full hover:bg-red-600/10 transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
                Выйти
              </button>
            </div>
          </div>

          {/* Profile Details */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-[#151515] border border-[#262626] rounded p-8">
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-2xl font-bold text-white">Информация о профиле</h3>
              </div>

              <div className="space-y-6">
                {/* Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Имя
                  </label>
                  <div className="flex items-center gap-3 p-4 bg-[#0a0a0a] border border-[#262626] rounded">
                    <User className="w-5 h-5 text-gray-500" />
                    <span className="text-white">{user.username}</span>
                  </div>
                </div>

                {/* Email */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Email
                  </label>
                  <div className="flex items-center gap-3 p-4 bg-[#0a0a0a] border border-[#262626] rounded">
                    <Mail className="w-5 h-5 text-gray-500" />
                    <span className="text-white">{user.email}</span>
                  </div>
                </div>

                {/* Role */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Роль
                  </label>
                  <div className="flex items-center gap-4 p-4 bg-[#0a0a0a] border border-[#262626] rounded">
                    <div className={`w-10 h-10 rounded bg-gradient-to-tr ${currentRole?.color} flex items-center justify-center`}>
                      {currentRole?.icon && <currentRole.icon className="w-5 h-5 text-white" />}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-white">{currentRole?.label}</div>
                      <div className="text-sm text-gray-400">{currentRole?.description}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

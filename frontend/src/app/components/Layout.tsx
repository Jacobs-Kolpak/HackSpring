import { Link, Outlet, useLocation } from 'react-router';
import { 
  Menu,
  X,
  User,
  LogIn
} from 'lucide-react';
import { useState } from 'react';
import { useDocuments } from '../context/DocumentContext';
import { useAuth } from '../context/AuthContext';
import { NAVIGATION_ITEMS } from '../constants/navigation';

export function Layout() {
  const location = useLocation();
  const { documents } = useDocuments();
  const { user } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-6 left-6 z-50">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-3 rounded-full bg-[#151515] border border-[#262626] hover:border-[#38C571] transition-all hover:shadow-lg hover:shadow-[#38C571]/20"
        >
          {sidebarOpen ? <X className="w-5 h-5 text-gray-400" /> : <Menu className="w-5 h-5 text-gray-400" />}
        </button>
      </div>

      {/* Sidebar - Icon Only */}
      <aside className={`
        fixed top-0 left-0 h-full w-20 z-40 transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="h-full bg-[#0f0f0f] border-r border-[#262626] flex flex-col relative overflow-hidden">
          {/* Futuristic glow effect */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#38C571] to-transparent opacity-30"></div>
          
          {/* Navigation */}
          <nav className="flex-1 p-3 space-y-2 overflow-y-auto">
            {NAVIGATION_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href;
              
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  title={item.name}
                  className={`
                    group flex items-center justify-center w-14 h-14 rounded transition-all relative overflow-hidden
                    ${isActive 
                      ? 'bg-gradient-to-br from-[#38C571] to-[#70D116] text-[#0a0a0a] shadow-lg shadow-[#38C571]/30' 
                      : 'text-gray-500 hover:text-gray-200 hover:bg-[#1a1a1a]'
                    }
                  `}
                >
                  {isActive && (
                    <>
                      <div className="absolute inset-0 bg-gradient-to-br from-[#38C571] to-[#70D116] opacity-20 blur-xl"></div>
                      <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-[#38C571] to-[#70D116] rounded-r"></div>
                    </>
                  )}
                  <Icon className={`w-6 h-6 relative z-10 ${isActive ? 'text-[#0a0a0a]' : 'group-hover:text-[#38C571]'}`} />
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-3 border-t border-[#262626]">
            {documents.length > 0 && (
              <div className="w-14 h-14 rounded bg-[#151515] border border-[#262626] flex items-center justify-center relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-[#38C571] to-transparent"></div>
                <div className="px-2 py-1 bg-gradient-to-br from-[#38C571] to-[#70D116] rounded-full shadow-lg shadow-[#38C571]/30">
                  <span className="text-xs font-bold text-[#0a0a0a]">{documents.length}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="lg:ml-20 min-h-screen">
        {/* Top Bar with User Profile */}
        <div className="fixed top-0 right-0 left-0 lg:left-20 z-20 bg-[#0f0f0f]/80 backdrop-blur-sm border-b border-[#262626]">
          <div className="flex items-center justify-end px-6 py-3">
            {user ? (
              <Link 
                to="/profile" 
                className="flex items-center gap-3 px-4 py-2 bg-[#151515] border border-[#262626] rounded-full hover:border-[#38C571] transition-all group"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#38C571] to-[#70D116] flex items-center justify-center shadow-lg shadow-[#38C571]/30">
                  <User className="w-4 h-4 text-white" />
                </div>
                <span className="text-sm text-gray-300 group-hover:text-white transition-colors hidden sm:block">
                  {user.username}
                </span>
              </Link>
            ) : (
              <Link 
                to="/login" 
                className="flex items-center gap-2 px-6 py-2 bg-gradient-to-tr from-[#38C571] to-[#70D116] text-[#0a0a0a] rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 transition-all font-medium"
              >
                <LogIn className="w-4 h-4" />
                <span className="hidden sm:inline">Войти</span>
              </Link>
            )}
          </div>
        </div>
        
        {/* Page content with top padding */}
        <div className="pt-16">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

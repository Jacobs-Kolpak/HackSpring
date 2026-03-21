import { RouterProvider } from 'react-router';
import { DocumentProvider } from './context/DocumentContext';
import { AuthProvider } from './context/AuthContext';
import { router } from './routes';

export default function App() {
  return (
    <AuthProvider>
      <DocumentProvider>
        <RouterProvider router={router} />
      </DocumentProvider>
    </AuthProvider>
  );
}
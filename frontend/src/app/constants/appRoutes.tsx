import { Layout } from '../components/Layout';
import Home from '../pages/Home';
import AudioPage from '../pages/AudioPage';
import ReportsPage from '../pages/ReportsPage';
import MindmapPage from '../pages/MindmapPage';
import FlashcardsPage from '../pages/FlashcardsPage';
import DataTablePage from '../pages/DataTablePage';
import VideoPage from '../pages/VideoPage';
import InfographicsPage from '../pages/InfographicsPage';
import PresentationPage from '../pages/PresentationPage';
import ProfilePage from '../pages/ProfilePage';

export const PROTECTED_ROUTES_CHILDREN = [
  {
    path: '/',
    Component: Layout,
    children: [
      { index: true, Component: Home },
      { path: 'profile', Component: ProfilePage },
      { path: 'audio', Component: AudioPage },
      { path: 'reports', Component: ReportsPage },
      { path: 'mindmap', Component: MindmapPage },
      { path: 'flashcards', Component: FlashcardsPage },
      { path: 'data-table', Component: DataTablePage },
      { path: 'video', Component: VideoPage },
      { path: 'infographics', Component: InfographicsPage },
      { path: 'presentation', Component: PresentationPage },
    ],
  },
];

import {
  BarChart3,
  BookOpen,
  FileText,
  Home,
  Mic,
  Network,
  Presentation,
  Table,
  Video,
  type LucideIcon,
} from "lucide-react";

export interface NavigationItem {
  name: string;
  href: string;
  icon: LucideIcon;
}

export const NAVIGATION_ITEMS: NavigationItem[] = [
  { name: "Главная", href: "/", icon: Home },
  { name: "Аудиопересказ", href: "/audio", icon: Mic },
  { name: "Отчеты", href: "/reports", icon: FileText },
  { name: "Интеллект-карта", href: "/mindmap", icon: Network },
  { name: "Карточки и Тесты", href: "/flashcards", icon: BookOpen },
  { name: "Таблица данных", href: "/data-table", icon: Table },
  { name: "Видео пересказ", href: "/video", icon: Video },
  { name: "Инфографика", href: "/infographics", icon: BarChart3 },
  { name: "Презентация", href: "/presentation", icon: Presentation },
];

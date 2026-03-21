import {
  ScrollText,
  Sparkles,
  Wand2,
  type LucideIcon,
} from "lucide-react";

export type SummaryModeId = "official" | "free" | "custom";

export interface SummaryMode {
  id: SummaryModeId;
  name: string;
  description: string;
  accent: string;
  icon: LucideIcon;
}

export const SUMMARY_MODES: SummaryMode[] = [
  {
    id: "official",
    name: "Официальное саммари",
    description:
      "Строгий деловой тон, четкая структура и акцент на ключевых выводах.",
    accent: "#8b5cf6",
    icon: ScrollText,
  },
  {
    id: "free",
    name: "Свободное саммари",
    description:
      "Более живой и простой стиль, чтобы быстро понять суть документа.",
    accent: "#22c55e",
    icon: Sparkles,
  },
  {
    id: "custom",
    name: "Кастомный шаблон",
    description:
      "Пользователь задает собственный шаблон, стиль и дополнительные инструкции.",
    accent: "#f97316",
    icon: Wand2,
  },
];

export const DEFAULT_CUSTOM_TEMPLATE =
  "Сделай суммаризацию документа в удобном и полезном виде.";

export const DEFAULT_CUSTOM_SYSTEM_PROMPT =
  "Ты эксперт по анализу документов. Пиши ясно, точно и без выдуманных фактов. Используй только информацию из файла.";

export const DEFAULT_CUSTOM_STYLE =
  "Понятный, дружелюбный и естественный стиль без сложных формулировок.";

export const DEFAULT_CUSTOM_FOCUS =
  "Ключевые тезисы, главные выводы и важные детали.";

export const DEFAULT_CUSTOM_FORMAT =
  "Сначала короткий общий вывод, затем структурированное summary.";

export const REPORTS_PAGE_STORAGE_KEY = "reports_page_state_v1";

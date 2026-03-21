import { Trash2 } from "lucide-react";

interface PageClearButtonProps {
  disabled?: boolean;
  onClick: () => void;
  title?: string;
}

export default function PageClearButton({
  disabled = false,
  onClick,
  title = "Очистить вывод",
}: PageClearButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      className="fixed right-4 bottom-4 z-50 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-[#111111]/70 text-gray-500 backdrop-blur-sm transition-all hover:border-white/20 hover:text-gray-200 disabled:cursor-default disabled:opacity-30 disabled:hover:border-white/10 disabled:hover:text-gray-500"
    >
      <Trash2 className="h-4 w-4" />
    </button>
  );
}

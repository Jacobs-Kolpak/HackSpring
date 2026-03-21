import { useEffect, useRef, useState } from "react";
import {
  Video,
  Play,
  Pause,
  Volume2,
  Download,
  Lightbulb,
} from "lucide-react";
import { useDocuments } from "../context/DocumentContext";
import PageClearButton from "../components/PageClearButton";

export default function VideoPage() {
  const { documents } = useDocuments();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generated, setGenerated] = useState(false);
  const [duration, setDuration] = useState(5);
  const [style, setStyle] = useState<
    "modern" | "classic" | "minimal"
  >("modern");
  const generationTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (generationTimeoutRef.current !== null) {
        window.clearTimeout(generationTimeoutRef.current);
      }
    };
  }, []);

  const handleGenerate = () => {
    setIsGenerating(true);
    if (generationTimeoutRef.current !== null) {
      window.clearTimeout(generationTimeoutRef.current);
    }
    generationTimeoutRef.current = window.setTimeout(() => {
      setIsGenerating(false);
      setGenerated(true);
      generationTimeoutRef.current = null;
    }, 3000);
  };

  const hasClearableOutput = generated || isGenerating;

  const handleClearPageOutput = () => {
    if (generationTimeoutRef.current !== null) {
      window.clearTimeout(generationTimeoutRef.current);
      generationTimeoutRef.current = null;
    }
    setIsGenerating(false);
    setGenerated(false);
  };

  if (documents.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-md">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#84cc16] flex items-center justify-center shadow-2xl shadow-[#84cc16]/20">
            <Video className="w-10 h-10 text-[#84cc16]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы для создания видео-пересказа
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 lg:p-12">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput}
      />
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#84cc16] rounded-full mb-6">
            <Lightbulb className="w-4 h-4 text-[#84cc16]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
              AI Video Synthesis
            </span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Видеопересказ
          </h1>
          <p className="text-lg text-gray-400">
            Создание видео с изображениями и звуковой дорожкой
            по материалам
          </p>
        </div>

        {!generated ? (
          <div className="bg-[#151515] border border-[#262626] rounded p-8 relative overflow-hidden">
            {/* Futuristic accent lines */}
            <div className="absolute top-0 right-0 w-40 h-40 opacity-10">
              <div className="absolute top-0 right-0 w-32 h-0.5 bg-gradient-to-l from-[#84cc16] to-transparent"></div>
              <div className="absolute top-0 right-0 w-0.5 h-32 bg-gradient-to-b from-[#84cc16] to-transparent"></div>
            </div>

            {/* Settings */}
            <div className="mb-8 relative z-10">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded bg-gradient-to-br from-[#84cc16] to-[#65a30d] flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-semibold text-white">
                  Настройки видео
                </h2>
              </div>

              {/* Duration */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Длительность видео:{" "}
                  <span className="text-[#84cc16]">
                    {duration} минут
                  </span>
                </label>
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="1"
                  value={duration}
                  onChange={(e) =>
                    setDuration(parseInt(e.target.value))
                  }
                  className="w-full h-2 bg-[#262626] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-br [&::-webkit-slider-thumb]:from-[#84cc16] [&::-webkit-slider-thumb]:to-[#65a30d] [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-[#84cc16]/50"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>3 мин</span>
                  <span>15 мин</span>
                </div>
              </div>

              {/* Style */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Стиль оформления
                </label>
                <div className="grid md:grid-cols-3 gap-4">
                  <button
                    onClick={() => setStyle("modern")}
                    className={`p-5 rounded border-2 transition-all ${
                      style === "modern"
                        ? "border-[#84cc16] bg-gradient-to-br from-[#84cc16]/10 to-[#65a30d]/10"
                        : "border-[#262626] hover:border-[#404040] bg-[#1a1a1a]"
                    }`}
                  >
                    <div className="font-medium text-white">
                      Современный
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                      Динамичные анимации
                    </div>
                  </button>

                  <button
                    onClick={() => setStyle("classic")}
                    className={`p-5 rounded border-2 transition-all ${
                      style === "classic"
                        ? "border-[#84cc16] bg-gradient-to-br from-[#84cc16]/10 to-[#65a30d]/10"
                        : "border-[#262626] hover:border-[#404040] bg-[#1a1a1a]"
                    }`}
                  >
                    <div className="font-medium text-white">
                      Классический
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                      Плавные переходы
                    </div>
                  </button>

                  <button
                    onClick={() => setStyle("minimal")}
                    className={`p-5 rounded border-2 transition-all ${
                      style === "minimal"
                        ? "border-[#84cc16] bg-gradient-to-br from-[#84cc16]/10 to-[#65a30d]/10"
                        : "border-[#262626] hover:border-[#404040] bg-[#1a1a1a]"
                    }`}
                  >
                    <div className="font-medium text-white">
                      Минималистичный
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                      Простота и чистота
                    </div>
                  </button>
                </div>
              </div>

              {/* Documents */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Источники ({documents.length})
                </label>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      className="flex items-center gap-3 p-3 bg-[#1a1a1a] rounded border border-[#262626]"
                    >
                      <div className="w-2 h-2 bg-[#22c55e] rounded-full"></div>
                      <span className="text-sm text-gray-300 truncate">
                        {doc.name}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="w-full px-8 py-4 bg-gradient-to-r from-[#84cc16] to-[#65a30d] text-white rounded-full hover:shadow-2xl hover:shadow-[#84cc16]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3 font-semibold text-lg"
            >
              <Video className="w-6 h-6" />
              {isGenerating
                ? "Генерация видео..."
                : "Сгенерировать видео"}
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Video Preview */}
            <div className="bg-gradient-to-br from-[#84cc16] via-[#65a30d] to-[#4d7c0f] rounded border-2 border-[#84cc16] shadow-2xl shadow-[#84cc16]/30 p-2 relative overflow-hidden">
              {/* Background pattern */}
              <div className="absolute inset-0 opacity-5">
                <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
              </div>

              <div className="relative z-10 bg-black/80 aspect-video rounded flex items-center justify-center">
                <button className="w-24 h-24 bg-white/20 backdrop-blur-xl rounded-full flex items-center justify-center border-2 border-white/30 hover:bg-white/30 transition-all hover:scale-110">
                  <Play className="w-12 h-12 text-white ml-2" />
                </button>
              </div>
            </div>

            {/* Controls */}
            <div className="bg-[#151515] border border-[#262626] rounded p-6">
              <div className="flex flex-wrap gap-4">
                <button className="flex-1 px-6 py-3 bg-gradient-to-r from-[#84cc16] to-[#65a30d] text-white rounded-full hover:shadow-xl hover:shadow-[#84cc16]/30 transition-all flex items-center justify-center gap-2 font-medium">
                  <Play className="w-5 h-5" />
                  Воспроизвести
                </button>
                <button className="flex-1 px-6 py-3 bg-[#1a1a1a] border border-[#262626] text-gray-300 rounded-full hover:border-[#404040] transition-all flex items-center justify-center gap-2 font-medium">
                  <Download className="w-5 h-5" />
                  Скачать MP4
                </button>
              </div>

              {/* Video Info */}
              <div className="mt-6 pt-6 border-t border-[#262626] grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-[#84cc16]">
                    {duration}:00
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Длительность
                  </div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-[#84cc16]">
                    1080p
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Качество
                  </div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-[#84cc16] capitalize">
                    {style}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Стиль
                  </div>
                </div>
              </div>
            </div>

            {/* Info */}
            <div className="p-6 bg-[#151515] border border-[#84cc16]/30 rounded">
              <p className="text-sm text-gray-300 leading-relaxed flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-[#84cc16] flex-shrink-0 mt-0.5" />
                <span>
                  <strong className="text-white">Совет:</strong>{" "}
                  Видео-пересказ создает динамическое видео с
                  автоматически подобранными изображениями,
                  анимациями и озвученным текстом на основе
                  ваших документов. Отлично подходит для
                  презентаций и обучающих материалов.
                </span>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

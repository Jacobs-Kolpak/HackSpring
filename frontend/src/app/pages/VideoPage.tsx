import { useEffect, useMemo, useState } from "react";
import {
  Video,
  Download,
  Lightbulb,
  Loader2,
  AlertCircle,
  House,
  Sparkles,
} from "lucide-react";
import axios from "axios";
import { useDocuments } from "../context/DocumentContext";
import PageClearButton from "../components/PageClearButton";
import { useNavigate } from "react-router-dom";
import { API_ROUTES } from "../constants/api";
import {
  readSessionState,
  writeSessionState,
} from "../utils/sessionState";

interface GeneratedVideo {
  url: string;
  filename: string;
  sourceFileName: string;
  speaker: string;
  summaryTemplate: string;
  maxSentences: number;
}

const VIDEO_PAGE_STORAGE_KEY = "video_page_state_v1";

const DEFAULT_SPEAKER = "xenia";
const DEFAULT_SUMMARY_TEMPLATE = "summary";
const DEFAULT_MAX_SENTENCES = 8;
const SPEAKER_OPTIONS = [
  "aidar",
  "baya",
  "eugene",
  "kseniya",
  "xenia",
] as const;
const SUMMARY_TEMPLATE_OPTIONS = [
  "summary",
  "executive",
  "detailed",
] as const;

const parseErrorBlob = async (blob: Blob): Promise<string | null> => {
  try {
    const text = await blob.text();
    if (!text.trim()) {
      return null;
    }

    try {
      const parsed = JSON.parse(text) as {
        detail?: string;
        message?: string;
      };
      if (typeof parsed.detail === "string") {
        return parsed.detail;
      }
      if (typeof parsed.message === "string") {
        return parsed.message;
      }
    } catch {
      return text;
    }

    return text;
  } catch {
    return null;
  }
};

const extractFilename = (headerValue?: string) => {
  if (!headerValue) {
    return "video-summary.mp4";
  }

  const utf8Match = headerValue.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1]);
  }

  const asciiMatch = headerValue.match(/filename="?([^";]+)"?/i);
  if (asciiMatch?.[1]) {
    return asciiMatch[1];
  }

  return "video-summary.mp4";
};

export default function VideoPage() {
  const navigate = useNavigate();
  const { documents, uploadedSourceFiles } = useDocuments();
  const persistedState = readSessionState(
    VIDEO_PAGE_STORAGE_KEY,
    {
      speaker: DEFAULT_SPEAKER,
      summaryTemplate: DEFAULT_SUMMARY_TEMPLATE,
      maxSentences: DEFAULT_MAX_SENTENCES,
      selectedFileName: "",
      generationError: null as string | null,
    },
  );

  const [isGenerating, setIsGenerating] = useState(false);
  const [speaker, setSpeaker] = useState(persistedState.speaker);
  const [summaryTemplate, setSummaryTemplate] = useState(
    persistedState.summaryTemplate,
  );
  const [maxSentences, setMaxSentences] = useState(
    persistedState.maxSentences,
  );
  const [selectedFileName, setSelectedFileName] = useState(
    persistedState.selectedFileName,
  );
  const [generatedVideo, setGeneratedVideo] =
    useState<GeneratedVideo | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(
    persistedState.generationError,
  );

  const availableFiles = useMemo(
    () =>
      uploadedSourceFiles.filter((file) =>
        documents.some((doc) => doc.name === file.name),
      ),
    [documents, uploadedSourceFiles],
  );

  const selectedSourceFile = useMemo(() => {
    if (availableFiles.length === 0) {
      return null;
    }
    return (
      availableFiles.find(
        (file) => file.name === selectedFileName,
      ) || availableFiles[0]
    );
  }, [availableFiles, selectedFileName]);

  useEffect(() => {
    if (availableFiles.length > 0 && !selectedFileName) {
      setSelectedFileName(availableFiles[0].name);
    }
  }, [availableFiles, selectedFileName]);

  useEffect(() => {
    writeSessionState(VIDEO_PAGE_STORAGE_KEY, {
      speaker,
      summaryTemplate,
      maxSentences,
      selectedFileName,
      generationError,
    });
  }, [
    generationError,
    maxSentences,
    selectedFileName,
    speaker,
    summaryTemplate,
  ]);

  useEffect(() => {
    return () => {
      if (generatedVideo?.url) {
        URL.revokeObjectURL(generatedVideo.url);
      }
    };
  }, [generatedVideo]);

  const handleGenerate = async () => {
    if (!selectedSourceFile) {
      alert(
        "Исходный файл недоступен в памяти браузера. Загрузите документ заново на главной странице.",
      );
      return;
    }

    setIsGenerating(true);
    setGenerationError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedSourceFile);
      formData.append("speaker", speaker.trim() || DEFAULT_SPEAKER);
      formData.append(
        "summary_template",
        summaryTemplate || DEFAULT_SUMMARY_TEMPLATE,
      );
      formData.append(
        "max_sentences",
        String(
          Math.max(1, Math.min(20, Number(maxSentences) || DEFAULT_MAX_SENTENCES)),
        ),
      );

      const response = await axios.post(
        API_ROUTES.VIDEO.GENERATE,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        },
      );

      const contentType = response.headers["content-type"] || "";
      if (!contentType.includes("video/")) {
        const unexpectedMessage = await parseErrorBlob(
          response.data,
        );
        throw new Error(
          unexpectedMessage ||
            "Сервис вернул неожиданный формат ответа вместо видео.",
        );
      }

      if (generatedVideo?.url) {
        URL.revokeObjectURL(generatedVideo.url);
      }

      const objectUrl = URL.createObjectURL(response.data);
      setGeneratedVideo({
        url: objectUrl,
        filename: extractFilename(
          response.headers["content-disposition"],
        ),
        sourceFileName: selectedSourceFile.name,
        speaker: speaker.trim() || DEFAULT_SPEAKER,
        summaryTemplate:
          summaryTemplate || DEFAULT_SUMMARY_TEMPLATE,
        maxSentences: Math.max(
          1,
          Math.min(20, Number(maxSentences) || DEFAULT_MAX_SENTENCES),
        ),
      });
    } catch (error) {
      let errorMessage =
        "Не удалось сгенерировать видео. Проверьте параметры и попробуйте снова.";

      if (axios.isAxiosError(error)) {
        if (error.response?.data instanceof Blob) {
          const blobMessage = await parseErrorBlob(
            error.response.data,
          );
          if (blobMessage) {
            errorMessage = blobMessage;
          }
        } else if (typeof error.message === "string") {
          errorMessage = error.message;
        }
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      setGenerationError(errorMessage);
      alert(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!generatedVideo) {
      return;
    }

    const link = document.createElement("a");
    link.href = generatedVideo.url;
    link.download = generatedVideo.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const hasClearableOutput =
    Boolean(generatedVideo) ||
    Boolean(generationError) ||
    isGenerating;

  const handleClearPageOutput = () => {
    if (generatedVideo?.url) {
      URL.revokeObjectURL(generatedVideo.url);
    }

    setIsGenerating(false);
    setGeneratedVideo(null);
    setGenerationError(null);
  };

  if (documents.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6 bg-[#0a0a0a]">
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
          <button
            onClick={() => navigate("/")}
            className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#84cc16] to-[#65a30d] rounded-full text-[#0a0a0a] font-semibold"
          >
            <House className="w-4 h-4" />
            На главную
          </button>
        </div>
      </div>
    );
  }

  if (availableFiles.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6 bg-[#0a0a0a]">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-xl">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#84cc16] flex items-center justify-center shadow-2xl shadow-[#84cc16]/20">
            <AlertCircle className="w-10 h-10 text-[#84cc16]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет исходных файлов для отправки
          </h2>
          <p className="text-gray-400 text-lg">
            В памяти браузера остались карточки документов, но сами
            `File`-объекты недоступны. Загрузите документы заново на
            главной странице и вернитесь сюда.
          </p>
          <button
            onClick={() => navigate("/")}
            className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#84cc16] to-[#65a30d] rounded-full text-[#0a0a0a] font-semibold"
          >
            <House className="w-4 h-4" />
            Перезагрузить источник
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 lg:p-12 bg-[#0a0a0a] text-gray-300">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || isGenerating}
      />
      <div className="max-w-5xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#84cc16] rounded-full mb-6">
            <Sparkles className="w-4 h-4 text-[#84cc16]" />
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

        {!generatedVideo ? (
          <div className="bg-[#151515] border border-[#262626] rounded p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-40 h-40 opacity-10">
              <div className="absolute top-0 right-0 w-32 h-0.5 bg-gradient-to-l from-[#84cc16] to-transparent"></div>
              <div className="absolute top-0 right-0 w-0.5 h-32 bg-gradient-to-b from-[#84cc16] to-transparent"></div>
            </div>

            <div className="mb-8 relative z-10">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded bg-gradient-to-br from-[#84cc16] to-[#65a30d] flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-semibold text-white">
                  Настройки видео
                </h2>
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Исходный файл
                </label>
                <select
                  value={selectedSourceFile?.name || ""}
                  onChange={(e) =>
                    setSelectedFileName(e.target.value)
                  }
                  className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#84cc16]"
                >
                  {availableFiles.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Speaker
                </label>
                <select
                  value={speaker}
                  onChange={(e) =>
                    setSpeaker(e.target.value)
                  }
                  className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#262626] rounded text-white focus:outline-none focus:border-[#84cc16]"
                >
                  {SPEAKER_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Summary template
                  </label>
                  <select
                    value={summaryTemplate}
                    onChange={(e) =>
                      setSummaryTemplate(e.target.value)
                    }
                    className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#262626] rounded text-white focus:outline-none focus:border-[#84cc16]"
                  >
                    {SUMMARY_TEMPLATE_OPTIONS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Max sentences
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={maxSentences}
                    onChange={(e) =>
                      setMaxSentences(
                        Math.max(1, Number(e.target.value) || 1),
                      )
                    }
                    className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#262626] rounded text-white focus:outline-none focus:border-[#84cc16]"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Доступные источники ({availableFiles.length})
                </label>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {availableFiles.map((file) => (
                    <div
                      key={file.name}
                      className="flex items-center gap-3 p-3 bg-[#1a1a1a] rounded border border-[#262626]"
                    >
                      <div className="w-2 h-2 bg-[#22c55e] rounded-full"></div>
                      <span className="text-sm text-gray-300 truncate">
                        {file.name}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="w-full px-8 py-4 bg-gradient-to-r from-[#84cc16] to-[#65a30d] text-white rounded-full hover:shadow-2xl hover:shadow-[#84cc16]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3 font-semibold text-lg"
            >
              <Video className="w-6 h-6" />
              {isGenerating
                ? "Генерирую MP4..."
                : "Сгенерировать видео"}
            </button>

            {generationError && (
              <div className="mt-6 p-4 rounded border border-[#f97316]/30 bg-[#f97316]/10 text-sm text-orange-200">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{generationError}</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-[#84cc16] via-[#65a30d] to-[#4d7c0f] rounded border-2 border-[#84cc16] shadow-2xl shadow-[#84cc16]/30 p-2 relative overflow-hidden">
              <div className="absolute inset-0 opacity-5">
                <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
              </div>

              <video
                controls
                preload="metadata"
                src={generatedVideo.url}
                className="relative z-10 bg-black/80 aspect-video rounded w-full"
              >
                Ваш браузер не поддерживает элемент video.
              </video>
            </div>

            <div className="bg-[#151515] border border-[#262626] rounded p-6">
              <div className="flex flex-wrap gap-4">
                <button
                  onClick={handleDownload}
                  className="flex-1 px-6 py-3 bg-[#1a1a1a] border border-[#262626] text-gray-300 rounded-full hover:border-[#404040] transition-all flex items-center justify-center gap-2 font-medium"
                >
                  <Download className="w-5 h-5" />
                  Скачать MP4
                </button>
              </div>

              <div className="mt-6 pt-6 border-t border-[#262626] grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-[#84cc16]">
                    {generatedVideo.maxSentences}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Max sentences
                  </div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-[#84cc16]">
                    {generatedVideo.summaryTemplate}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Template
                  </div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-[#84cc16] capitalize">
                    {generatedVideo.speaker}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Speaker
                  </div>
                </div>
              </div>
            </div>

            <div className="p-6 bg-[#151515] border border-[#84cc16]/30 rounded">
              <div className="grid gap-3 md:grid-cols-2 mb-5 text-sm">
                <div className="rounded border border-[#262626] bg-[#101010] px-4 py-3">
                  <span className="text-gray-500">Файл:</span>{" "}
                  <span className="text-white">
                    {generatedVideo.sourceFileName}
                  </span>
                </div>
                <div className="rounded border border-[#262626] bg-[#101010] px-4 py-3">
                  <span className="text-gray-500">Speaker:</span>{" "}
                  <span className="text-white">
                    {generatedVideo.speaker}
                  </span>
                </div>
                <div className="rounded border border-[#262626] bg-[#101010] px-4 py-3">
                  <span className="text-gray-500">
                    Summary template:
                  </span>{" "}
                  <span className="text-white">
                    {generatedVideo.summaryTemplate}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {isGenerating && (
          <div className="mt-6 rounded border border-[#262626] bg-[#151515] px-5 py-4 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-[#84cc16]" />
            <span className="text-sm text-gray-300">
              Внешний сервис генерирует MP4. Это может занять время
              даже для короткого документа.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

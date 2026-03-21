import { useEffect, useMemo, useState } from "react";
import {
  Presentation as PresentationIcon,
  Download,
  Loader2,
  Sparkles,
  FileText,
  Zap,
} from "lucide-react";
import axios from "axios";
import axiosInstance from "../utils/axiosInstance";
import { API_BASE_URL, API_ROUTES } from "../constants/api";
import { useDocuments } from "../context/DocumentContext";
import PageClearButton from "../components/PageClearButton";
import {
  readSessionState,
  writeSessionState,
} from "../utils/sessionState";

interface PresentationResponse {
  title: string;
  slides_count: number;
  download_url: string;
  meta: Record<string, unknown>;
}

const PRESENTATION_PAGE_STORAGE_KEY =
  "presentation_page_state_v1";

export default function PresentationPage() {
  const { documents } = useDocuments();
  const persistedState = readSessionState(
    PRESENTATION_PAGE_STORAGE_KEY,
    {
      query: "",
      topK: 8,
      maxSlides: 8,
      selectedDocumentIds: [] as string[],
      generationError: null as string | null,
      generatedPresentation: null as PresentationResponse | null,
    },
  );
  const [query, setQuery] = useState(persistedState.query);
  const [topK, setTopK] = useState(persistedState.topK);
  const [maxSlides, setMaxSlides] = useState(
    persistedState.maxSlides,
  );
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>(
    persistedState.selectedDocumentIds,
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(
    persistedState.generationError,
  );
  const [generatedPresentation, setGeneratedPresentation] =
    useState<PresentationResponse | null>(
      persistedState.generatedPresentation,
    );

  const resolveDownloadUrl = (url: string) => {
    if (/^https?:\/\//i.test(url)) {
      return url;
    }

    const apiOrigin = new URL(API_BASE_URL).origin;
    return new URL(url, apiOrigin).toString();
  };

  const extractFilename = (downloadUrl: string) =>
    downloadUrl.split("/").pop()?.split("?")[0] || "presentation.pptx";

  useEffect(() => {
    if (
      documents.length > 0 &&
      selectedDocumentIds.length === 0
    ) {
      setSelectedDocumentIds(documents.map((doc) => doc.id));
    }
  }, [documents, selectedDocumentIds.length]);

  useEffect(() => {
    writeSessionState(PRESENTATION_PAGE_STORAGE_KEY, {
      query,
      topK,
      maxSlides,
      selectedDocumentIds,
      generationError,
      generatedPresentation,
    });
  }, [
    generatedPresentation,
    generationError,
    maxSlides,
    query,
    selectedDocumentIds,
    topK,
  ]);

  const selectedDocuments = useMemo(
    () =>
      documents.filter((doc) =>
        selectedDocumentIds.includes(doc.id),
      ),
    [documents, selectedDocumentIds],
  );

  const handleGenerate = async () => {
    if (!query.trim()) {
      alert("Введите тему или запрос для генерации презентации.");
      return;
    }

    if (selectedDocuments.length === 0) {
      alert("Выберите хотя бы один файл для генерации презентации.");
      return;
    }

    setIsGenerating(true);
    setGeneratedPresentation(null);
    setGenerationError(null);

    try {
      const scopedQuery = `Используй только следующие документы: ${selectedDocuments
        .map((doc) => doc.name)
        .join(", ")}. Запрос пользователя: ${query}`;

      const payload: Record<string, unknown> = {
        query: scopedQuery,
        collection: "docs_ci",
        top_k: topK,
        max_slides: maxSlides,
      };

      console.log("Presentation generate request:", payload);
      const response = await axios.post(
        API_ROUTES.PRESENTATION.GENERATE,
        payload,
      );

      console.log("Presentation generate response:", response.data);
      setGeneratedPresentation(response.data);
    } catch (error) {
      console.error("Error generating presentation:", error);
      let errorMessage =
        "Не удалось сгенерировать презентацию. Проверьте параметры и попробуйте снова.";

      if (axios.isAxiosError(error)) {
        console.error(
          "Presentation generate response data:",
          error.response?.data,
        );

        const detail =
          typeof error.response?.data?.detail === "string"
            ? error.response.data.detail
            : null;

        if (detail === "LLM response does not contain JSON") {
          errorMessage =
            "Не удалось собрать презентацию: модель не нашла достаточно структурированного контекста в загруженных данных. Попробуйте уточнить запрос или загрузить более релевантный материал.";
        } else if (detail) {
          errorMessage = `Не удалось сгенерировать презентацию: ${detail}`;
        }
      }

      setGenerationError(errorMessage);
      alert(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (!generatedPresentation?.download_url) return;

    setIsDownloading(true);

    try {
      const filename = extractFilename(generatedPresentation.download_url);
      const directDownloadUrl = `${API_ROUTES.PRESENTATION.DOWNLOAD}/${filename}`;
      console.log("Presentation download request:", {
        filename,
        directDownloadUrl,
      });
      const response = await axiosInstance.get(
        resolveDownloadUrl(directDownloadUrl),
        { responseType: "blob" },
      );
      console.log("Presentation download response:", {
        filename,
        size: response.data?.size,
        type: response.data?.type,
      });

      const blobUrl = URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error("Error downloading presentation:", error);
      alert("Не удалось скачать презентацию.");
    } finally {
      setIsDownloading(false);
    }
  };

  const hasClearableOutput =
    Boolean(generatedPresentation) || Boolean(generationError);

  const handleClearPageOutput = () => {
    setGeneratedPresentation(null);
    setGenerationError(null);
    writeSessionState(PRESENTATION_PAGE_STORAGE_KEY, {
      query,
      topK,
      maxSlides,
      selectedDocumentIds,
      generationError: null,
      generatedPresentation: null,
    });
  };

  if (documents.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-md">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#22c55e] flex items-center justify-center shadow-2xl shadow-[#22c55e]/20">
            <PresentationIcon className="w-10 h-10 text-[#22c55e]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы для создания презентации
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 lg:p-12">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || isGenerating || isDownloading}
      />
      <div className="max-w-6xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#22c55e] rounded-full mb-6">
            <Zap className="w-4 h-4 text-[#22c55e]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
              AI Presentation
            </span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Презентация
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl">
            Выберите документы, задайте тему и сгенерируйте презентацию по загруженным материалам.
          </p>
        </div>

        {generationError && (
          <div className="mb-6 rounded border border-[#f97316]/40 bg-[#2a1708] px-5 py-4 text-sm text-orange-200">
            {generationError}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Файлы для презентации
                </label>
                <div className="max-h-56 overflow-y-auto rounded border border-[#262626] bg-[#0f0f0f] p-3 space-y-2">
                  {documents.map((doc) => (
                    <label
                      key={doc.id}
                      className="flex items-center gap-3 px-3 py-2 rounded border border-transparent hover:border-[#22c55e]/30 transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedDocumentIds.includes(doc.id)}
                        onChange={(e) => {
                          setSelectedDocumentIds((prev) =>
                            e.target.checked
                              ? [...prev, doc.id]
                              : prev.filter((id) => id !== doc.id),
                          );
                        }}
                        className="accent-[#22c55e]"
                      />
                      <FileText className="w-4 h-4 text-[#22c55e]" />
                      <span className="text-sm text-gray-300">
                        {doc.name}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Запрос / тема презентации
                </label>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Например: Ключевые идеи загруженных документов"
                  className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#22c55e]"
                />
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Широта охвата контекста
                  </label>
                  <input
                    type="number"
                    min={1}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#22c55e]"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Максимум слайдов
                  </label>
                  <input
                    type="number"
                    min={1}
                    value={maxSlides}
                    onChange={(e) =>
                      setMaxSlides(Number(e.target.value))
                    }
                    className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#22c55e]"
                  />
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="mt-8 w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-[#22c55e] to-[#16a34a] text-white rounded-full hover:shadow-2xl hover:shadow-[#22c55e]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Генерируем презентацию...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Сгенерировать презентацию
                </>
              )}
            </button>
          </div>

          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Результат генерации
            </h2>

            {generatedPresentation ? (
              <div className="space-y-6">
                <div className="rounded border border-[#262626] bg-[#0f0f0f] p-5">
                  <p className="text-sm text-gray-500 mb-2">Название</p>
                  <h3 className="text-2xl font-semibold text-white">
                    {generatedPresentation.title}
                  </h3>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded border border-[#262626] bg-[#0f0f0f] p-5">
                    <p className="text-sm text-gray-500 mb-2">Слайдов</p>
                    <p className="text-3xl font-bold text-white">
                      {generatedPresentation.slides_count}
                    </p>
                  </div>
                  <div className="rounded border border-[#262626] bg-[#0f0f0f] p-5">
                    <p className="text-sm text-gray-500 mb-2">Download URL</p>
                    <p className="text-sm text-gray-300 break-all">
                      {generatedPresentation.download_url}
                    </p>
                  </div>
                </div>

                <div className="rounded border border-[#262626] bg-[#0f0f0f] p-5">
                  <p className="text-sm text-gray-500 mb-3">Meta</p>
                  <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                    {JSON.stringify(
                      generatedPresentation.meta || {},
                      null,
                      2,
                    )}
                  </pre>
                </div>

                <button
                  onClick={handleDownload}
                  disabled={isDownloading}
                  className="w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-[#22c55e] to-[#16a34a] text-white rounded-full hover:shadow-2xl hover:shadow-[#22c55e]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
                >
                  {isDownloading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Скачиваем...
                    </>
                  ) : (
                    <>
                      <Download className="w-5 h-5" />
                      Скачать презентацию
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="h-full min-h-[280px] rounded border border-dashed border-[#262626] bg-[#0f0f0f] flex flex-col items-center justify-center text-center p-8">
                <PresentationIcon className="w-12 h-12 text-[#22c55e] mb-4" />
                <p className="text-white text-lg font-medium mb-2">
                  Презентация еще не создана
                </p>
                <p className="text-gray-400 max-w-md">
                  Заполните параметры слева и запустите генерацию. После ответа backend здесь появятся метаданные и ссылка на скачивание.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

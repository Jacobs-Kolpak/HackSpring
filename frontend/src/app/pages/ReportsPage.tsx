import { useEffect, useMemo, useState } from "react";
import {
  Download,
  FileText,
  Loader2,
  Zap,
} from "lucide-react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { useDocuments } from "../context/DocumentContext";
import { API_ROUTES } from "../constants/api";
import PageClearButton from "../components/PageClearButton";
import {
  DEFAULT_CUSTOM_FOCUS,
  DEFAULT_CUSTOM_FORMAT,
  DEFAULT_CUSTOM_STYLE,
  DEFAULT_CUSTOM_SYSTEM_PROMPT,
  DEFAULT_CUSTOM_TEMPLATE,
  REPORTS_PAGE_STORAGE_KEY,
  SUMMARY_MODES,
  type SummaryModeId,
} from "../constants/reports";
import {
  readSessionState,
  writeSessionState,
} from "../utils/sessionState";
import axiosInstance from "../utils/axiosInstance";

interface SummaryResponse {
  summary: string;
  source: string;
  model: string;
  meta: Record<string, unknown>;
}

const escapeHtml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const markdownToPlainText = (value: string) =>
  value
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)")
    .replace(/^\s*[-*+]\s+/gm, "- ")
    .replace(/^\s*\d+\.\s+/gm, "")
    .trim();

export default function ReportsPage() {
  const { documents, uploadedSourceFiles } = useDocuments();
  const persistedState = readSessionState(
    REPORTS_PAGE_STORAGE_KEY,
    {
      selectedFileName: "",
      topic: "",
      maxSentences: 8,
      customSummaryTask: DEFAULT_CUSTOM_TEMPLATE,
      customStyle: DEFAULT_CUSTOM_STYLE,
      customFocus: DEFAULT_CUSTOM_FOCUS,
      customFormat: DEFAULT_CUSTOM_FORMAT,
      customSystemPrompt: DEFAULT_CUSTOM_SYSTEM_PROMPT,
      generatedSummary: null as SummaryResponse | null,
      selectedModeId: "official" as SummaryModeId,
      errorMessage: null as string | null,
    },
  );
  const [selectedFileName, setSelectedFileName] = useState<string>(
    persistedState.selectedFileName,
  );
  const [topic, setTopic] = useState(persistedState.topic);
  const [maxSentences, setMaxSentences] = useState<number>(
    persistedState.maxSentences,
  );
  const [customSummaryTask, setCustomSummaryTask] = useState(
    persistedState.customSummaryTask,
  );
  const [customStyle, setCustomStyle] = useState(
    persistedState.customStyle,
  );
  const [customFocus, setCustomFocus] = useState(
    persistedState.customFocus,
  );
  const [customFormat, setCustomFormat] = useState(
    persistedState.customFormat,
  );
  const [customSystemPrompt, setCustomSystemPrompt] = useState(
    persistedState.customSystemPrompt,
  );
  const [generatedSummary, setGeneratedSummary] = useState<
    SummaryResponse | null
  >(persistedState.generatedSummary);
  const [selectedModeId, setSelectedModeId] =
    useState<SummaryModeId>(persistedState.selectedModeId);
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(
    persistedState.errorMessage,
  );

  useEffect(() => {
    if (documents.length > 0 && !selectedFileName) {
      setSelectedFileName(documents[0].name);
    }
  }, [selectedFileName, documents]);

  useEffect(() => {
    writeSessionState(REPORTS_PAGE_STORAGE_KEY, {
      selectedFileName,
      topic,
      maxSentences,
      customSummaryTask,
      customStyle,
      customFocus,
      customFormat,
      customSystemPrompt,
      generatedSummary,
      selectedModeId,
      errorMessage,
    });
  }, [
    customFocus,
    customFormat,
    customStyle,
    customSummaryTask,
    customSystemPrompt,
    errorMessage,
    generatedSummary,
    maxSentences,
    selectedFileName,
    selectedModeId,
    topic,
  ]);

  // Find the selected document from the persisted documents list
  const selectedDoc = useMemo(() => {
    if (documents.length === 0) return null;
    return (
      documents.find((d) => d.name === selectedFileName) || documents[0]
    );
  }, [selectedFileName, documents]);

  // Also check uploadedSourceFiles for raw File objects (file uploads)
  const selectedFile = useMemo(() => {
    if (uploadedSourceFiles.length === 0) return null;
    return (
      uploadedSourceFiles.find((f) => f.name === selectedFileName) ||
      uploadedSourceFiles[0]
    );
  }, [selectedFileName, uploadedSourceFiles]);

  const availableFilesCount = documents.length;

  const selectedMode = SUMMARY_MODES.find(
    (mode) => mode.id === selectedModeId,
  ) || SUMMARY_MODES[0];

  const buildMarkdownExport = () => {
    if (!generatedSummary) {
      return "";
    }

    const sections = [
      "# Summary",
      "",
      `Источник: ${generatedSummary.source || selectedDoc?.name || selectedFile?.name || "Не указан"}`,
      `Модель: ${generatedSummary.model || "Не указана"}`,
      "",
      "## Summary Text",
      "",
      generatedSummary.summary,
    ];

    if (
      generatedSummary.meta &&
      Object.keys(generatedSummary.meta).length > 0
    ) {
      sections.push(
        "",
        "## Meta",
        "",
        "```json",
        JSON.stringify(generatedSummary.meta, null, 2),
        "```",
      );
    }

    return sections.join("\n");
  };

  const downloadFile = (
    filename: string,
    content: string,
    mimeType: string,
  ) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownload = (format: "md" | "txt" | "doc") => {
    if (!generatedSummary) {
      return;
    }

    const baseName =
      selectedFile?.name.replace(/\.[^.]+$/, "") || "summary";
    const markdownContent = buildMarkdownExport();

    if (format === "md") {
      downloadFile(
        `${baseName}-summary.md`,
        markdownContent,
        "text/markdown;charset=utf-8",
      );
      return;
    }

    if (format === "txt") {
      downloadFile(
        `${baseName}-summary.txt`,
        markdownToPlainText(markdownContent),
        "text/plain;charset=utf-8",
      );
      return;
    }

    const htmlDocument = `<!DOCTYPE html>
<html lang="ru">
  <head>
    <meta charset="UTF-8" />
    <title>Summary</title>
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.6; padding: 32px; color: #111; }
      h1, h2 { margin-bottom: 12px; }
      pre { white-space: pre-wrap; font-family: Consolas, monospace; background: #f5f5f5; padding: 16px; border-radius: 8px; }
      .meta { margin-bottom: 24px; }
    </style>
  </head>
  <body>
    <h1>Summary</h1>
    <div class="meta">
      <p><strong>Источник:</strong> ${escapeHtml(generatedSummary.source || selectedFile?.name || "Не указан")}</p>
      <p><strong>Модель:</strong> ${escapeHtml(generatedSummary.model || "Не указана")}</p>
    </div>
    <h2>Summary Text</h2>
    <pre>${escapeHtml(generatedSummary.summary)}</pre>
  </body>
</html>`;

    downloadFile(
      `${baseName}-summary.doc`,
      htmlDocument,
      "application/msword;charset=utf-8",
    );
  };

  const handleGenerateSummary = async () => {
    if (!selectedFile && !selectedDoc) {
      setErrorMessage(
        "Исходный файл не найден. Загрузите документ заново на главной странице.",
      );
      return;
    }

    if (
      selectedModeId === "custom" &&
      !customSummaryTask.trim() &&
      !customStyle.trim() &&
      !customFocus.trim() &&
      !customFormat.trim()
    ) {
      setErrorMessage(
        "Для кастомного саммари заполните хотя бы одну пользовательскую настройку.",
      );
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    setGeneratedSummary(null);

    const formData = new FormData();
    // Use raw File object if available, otherwise create Blob from persisted document content
    if (selectedFile) {
      formData.append("file", selectedFile);
    } else if (selectedDoc) {
      const blob = new Blob([selectedDoc.content], { type: selectedDoc.type || "text/plain" });
      formData.append("file", blob, selectedDoc.name);
    }

    if (topic.trim()) {
      formData.append("topic", topic.trim());
    }

    if (Number.isFinite(maxSentences) && maxSentences > 0) {
      formData.append("max_sentences", String(maxSentences));
    }

    if (selectedModeId === "official") {
      formData.append(
        "template",
        "Сделай официальное summary документа на русском языке. Используй деловой стиль, краткие формулировки, четкую структуру и акцент на ключевых выводах, фактах и рекомендациях.",
      );
    }

    if (selectedModeId === "free") {
      formData.append(
        "template",
        "Сделай summary документа на русском языке в свободном и понятном стиле. Объясни суть простыми словами, но сохрани важные факты, выводы и контекст.",
      );
    }

    if (selectedModeId === "custom") {
      const customTemplate = [
        `Суммаризация: ${customSummaryTask.trim() || "Сделай краткое и полезное summary документа."}`,
        `Стиль: ${customStyle.trim() || "Нейтральный, понятный и аккуратный."}`,
        `Что включить: ${customFocus.trim() || "Только самое важное без второстепенных деталей."}`,
        `Формат ответа: ${customFormat.trim() || "Связный, легко читаемый текст."}`,
      ].join("\n");

      formData.append("template", customTemplate);

      if (customSystemPrompt.trim()) {
        formData.append(
          "system_prompt",
          customSystemPrompt.trim(),
        );
      }
    }

    try {
      const response = await axiosInstance.post<SummaryResponse>(
        API_ROUTES.SUMMARY.FILE,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

      setGeneratedSummary(response.data);
    } catch (error) {
      let message =
        "Не удалось получить summary. Проверьте настройки и попробуйте снова.";

      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail;

        if (typeof detail === "string") {
          message = detail;
        } else if (Array.isArray(detail) && detail.length > 0) {
          message = detail
            .map((item) => item?.msg)
            .filter(Boolean)
            .join(", ");
        }
      }

      setErrorMessage(message);
    } finally {
      setIsGenerating(false);
    }
  };

  const hasClearableOutput =
    Boolean(generatedSummary) || Boolean(errorMessage);

  const handleClearPageOutput = () => {
    setGeneratedSummary(null);
    setErrorMessage(null);
    writeSessionState(REPORTS_PAGE_STORAGE_KEY, {
      selectedFileName,
      topic,
      maxSentences,
      customSummaryTask,
      customStyle,
      customFocus,
      customFormat,
      customSystemPrompt,
      generatedSummary: null,
      selectedModeId,
      errorMessage: null,
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
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#8b5cf6] flex items-center justify-center shadow-2xl shadow-[#8b5cf6]/20">
            <FileText className="w-10 h-10 text-[#8b5cf6]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы на главной странице для создания summary
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 lg:p-12">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || isGenerating}
      />
      <div className="max-w-6xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#8b5cf6] rounded-full mb-6">
            <Zap className="w-4 h-4 text-[#8b5cf6]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
              AI Summary
            </span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Summary
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl">
            Два готовых шаблона и один настраиваемый режим для summary по файлу.
          </p>
        </div>

        {errorMessage && (
          <div className="mb-6 rounded border border-[#ef4444]/40 bg-[#2a1013] px-5 py-4 text-sm text-red-200">
            {errorMessage}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-[1fr_0.95fr]">
          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Файл для summary
                </label>
                <select
                  value={selectedFile?.name || ""}
                  onChange={(e) =>
                    setSelectedFileName(e.target.value)
                  }
                  className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#8b5cf6]"
                >
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.name}>
                      {doc.name}
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-xs text-gray-500">
                  Endpoint работает с одним файлом за запрос. Доступно файлов:{" "}
                  {availableFilesCount}.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Тема summary
                </label>
                <input
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="Например: ключевые выводы для руководителя"
                  className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 focus:outline-none focus:border-[#8b5cf6]"
                />
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                {SUMMARY_MODES.map((mode) => {
                  const Icon = mode.icon;
                  const isSelected = selectedModeId === mode.id;

                  return (
                    <button
                      key={mode.id}
                      type="button"
                      onClick={() => setSelectedModeId(mode.id)}
                      className={`rounded border px-5 py-5 text-left transition-all ${
                        isSelected
                          ? "border-white/30 bg-[#101010] shadow-[0_0_0_1px_rgba(255,255,255,0.05)]"
                          : "border-[#262626] bg-[#0f0f0f] hover:border-white/20"
                      }`}
                    >
                      <div
                        className="w-11 h-11 rounded mb-4 flex items-center justify-center"
                        style={{
                          backgroundColor: `${mode.accent}20`,
                          border: `1px solid ${mode.accent}55`,
                        }}
                      >
                        <Icon
                          className="w-5 h-5"
                          style={{ color: mode.accent }}
                        />
                      </div>
                      <div className="text-white font-semibold mb-2">
                        {mode.name}
                      </div>
                      <div className="text-sm text-gray-400">
                        {mode.description}
                      </div>
                    </button>
                  );
                })}
              </div>

              {selectedModeId === "custom" && (
                <div className="rounded border border-[#f97316]/30 bg-[#1a120d] p-5 space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-white">
                        Настройка кастомного шаблона
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        Эти поля используются только для третьего режима.
                      </div>
                    </div>
                    <div className="text-xs text-[#f97316]">
                      Кастомный режим
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Суммаризация
                    </label>
                    <textarea
                      value={customSummaryTask}
                      onChange={(e) =>
                        setCustomSummaryTask(e.target.value)
                      }
                      rows={3}
                      placeholder="Например: сделай summary для подготовки к экзамену или встречи"
                      className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Стиль
                    </label>
                    <textarea
                      value={customStyle}
                      onChange={(e) =>
                        setCustomStyle(e.target.value)
                      }
                      rows={3}
                      placeholder="Например: простой, деловой, дружелюбный, академический"
                      className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Что включить
                    </label>
                    <textarea
                      value={customFocus}
                      onChange={(e) =>
                        setCustomFocus(e.target.value)
                      }
                      rows={3}
                      placeholder="Например: основные выводы, риски, цифры, рекомендации"
                      className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Формат ответа
                    </label>
                    <textarea
                      value={customFormat}
                      onChange={(e) =>
                        setCustomFormat(e.target.value)
                      }
                      rows={3}
                      placeholder="Например: короткое вступление и список из 5 пунктов"
                      className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      System prompt
                    </label>
                    <textarea
                      value={customSystemPrompt}
                      onChange={(e) =>
                        setCustomSystemPrompt(e.target.value)
                      }
                      rows={4}
                      placeholder="Дополнительные инструкции для модели"
                      className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                    />
                  </div>
                </div>
              )}

              <button
                type="button"
                onClick={handleGenerateSummary}
                disabled={isGenerating}
                className="w-full inline-flex items-center justify-center gap-3 rounded bg-white px-6 py-4 text-base font-semibold text-black transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
              >
                {isGenerating ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <selectedMode.icon className="w-5 h-5" />
                )}
                <span>
                  {isGenerating
                    ? "Генерируем summary..."
                    : `Сгенерировать: ${selectedMode.name}`}
                </span>
              </button>
            </div>
          </div>

          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <h2 className="text-2xl font-semibold text-white">
                  Результат summary
                </h2>
                <p className="text-sm text-gray-400 mt-1">
                  Источник, модель и итоговый текст ответа.
                </p>
              </div>
              {generatedSummary?.model && (
                <div className="px-3 py-2 rounded-full border border-[#262626] bg-[#0f0f0f] text-xs text-gray-300">
                  {generatedSummary.model}
                </div>
              )}
            </div>

            {generatedSummary && (
              <div className="flex flex-wrap gap-3 mb-6">
                <button
                  type="button"
                  onClick={() => handleDownload("md")}
                  className="inline-flex items-center gap-2 rounded border border-[#262626] bg-[#0f0f0f] px-4 py-2 text-sm font-medium text-gray-200 transition hover:border-[#8b5cf6]"
                >
                  <Download className="w-4 h-4" />
                  Скачать .md
                </button>
                <button
                  type="button"
                  onClick={() => handleDownload("txt")}
                  className="inline-flex items-center gap-2 rounded border border-[#262626] bg-[#0f0f0f] px-4 py-2 text-sm font-medium text-gray-200 transition hover:border-[#8b5cf6]"
                >
                  <Download className="w-4 h-4" />
                  Скачать .txt
                </button>
                <button
                  type="button"
                  onClick={() => handleDownload("doc")}
                  className="inline-flex items-center gap-2 rounded border border-[#262626] bg-[#0f0f0f] px-4 py-2 text-sm font-medium text-gray-200 transition hover:border-[#8b5cf6]"
                >
                  <Download className="w-4 h-4" />
                  Скачать .doc
                </button>
              </div>
            )}

            {generatedSummary ? (
              <div className="space-y-5">
                <div className="rounded border border-[#262626] bg-[#0f0f0f] p-4">
                  <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                    Source
                  </div>
                  <div className="text-sm text-white">
                    {generatedSummary.source || selectedFile?.name}
                  </div>
                </div>

                <div className="rounded border border-[#262626] bg-[#0f0f0f] p-5">
                  <div className="text-xs uppercase tracking-wider text-gray-500 mb-3">
                    Summary Text
                  </div>
                  <div className="prose prose-invert max-w-none prose-p:text-gray-200 prose-headings:text-white prose-strong:text-white prose-li:text-gray-200 prose-code:text-[#f5f5f5]">
                    <ReactMarkdown>
                      {generatedSummary.summary}
                    </ReactMarkdown>
                  </div>
                </div>

                {generatedSummary.meta &&
                  Object.keys(generatedSummary.meta).length > 0 && (
                    <div className="rounded border border-[#262626] bg-[#0f0f0f] p-4">
                      <div className="text-xs uppercase tracking-wider text-gray-500 mb-3">
                        Meta
                      </div>
                      <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                        {JSON.stringify(
                          generatedSummary.meta,
                          null,
                          2,
                        )}
                      </pre>
                    </div>
                  )}
              </div>
            ) : (
              <div className="min-h-[420px] rounded border border-dashed border-[#262626] bg-[#0f0f0f] flex items-center justify-center p-8 text-center">
                <div>
                  <div className="w-16 h-16 mx-auto mb-5 rounded-full bg-[#8b5cf6]/10 border border-[#8b5cf6]/20 flex items-center justify-center">
                    <FileText className="w-8 h-8 text-[#8b5cf6]" />
                  </div>
                  <p className="text-white font-medium mb-2">
                    Summary пока не сгенерирован
                  </p>
                  <p className="text-sm text-gray-400 max-w-sm">
                    Нажмите одну из трех кнопок слева: официальный,
                    свободный или кастомный режим.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

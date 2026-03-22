import { useState, useRef, useEffect } from "react";
import {
  Upload,
  FileText,
  File,
  Trash2,
  Send,
  Loader2,
  Shield,
  Zap,
  Database,
  BarChart3,
  Mic,
  FileText as FileTextIcon,
  Network,
  BookOpen,
  Table,
  Video,
  Presentation,
  Link as LinkIcon,
  Globe,
} from "lucide-react";
import { useDocuments } from "../context/DocumentContext";
import { useNavigate } from "react-router";
import ReactMarkdown from "react-markdown";
import axiosInstance from "../utils/axiosInstance";
import axios from "axios";
import { API_ROUTES } from "../constants/api";
import PageClearButton from "../components/PageClearButton";
import {
  readSessionState,
  writeSessionState,
} from "../utils/sessionState";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: MessageSource[];
}

interface SourceReference {
  source_name: string;
  chunk_index?: number;
  page_number?: number | null;
  page_label?: string | null;
}

type MessageSource = string | SourceReference;

const HOME_PAGE_STORAGE_KEY = "home_page_state_v1";

function normalizeSourceReference(
  source: MessageSource,
): SourceReference {
  if (typeof source === "string") {
    return { source_name: source };
  }
  return source;
}

function formatSourceReference(
  source: MessageSource,
): string {
  const normalized = normalizeSourceReference(source);
  const parts = [normalized.source_name];

  if (normalized.page_label) {
    parts.push(`стр. ${normalized.page_label}`);
  } else if (typeof normalized.page_number === "number") {
    parts.push(`стр. ${normalized.page_number}`);
  } else if (
    typeof normalized.chunk_index === "number" &&
    normalized.chunk_index >= 0
  ) {
    parts.push(`чанк ${normalized.chunk_index + 1}`);
  }

  return parts.join(" • ");
}

function buildMessageSources(results: any[]): SourceReference[] {
  const seen = new Set<string>();
  const collected: SourceReference[] = [];

  for (const item of results) {
    if (!item || typeof item.source_name !== "string") {
      continue;
    }

    const reference: SourceReference = {
      source_name: item.source_name,
      chunk_index:
        typeof item.chunk_index === "number"
          ? item.chunk_index
          : undefined,
      page_number:
        typeof item.page_number === "number"
          ? item.page_number
          : null,
      page_label:
        typeof item.page_label === "string"
          ? item.page_label
          : null,
    };

    const dedupeKey = [
      reference.source_name,
      reference.page_label ?? reference.page_number ?? "",
      reference.page_label || reference.page_number ? "" : reference.chunk_index ?? "",
    ].join("::");

    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    collected.push(reference);
  }

  return collected;
}

export default function Home() {
  const {
    currentMindmapGraphData,
    currentPodcastAudioLoading,
    currentPodcastAudioUrl,
    currentFlashcards,
    currentTests,
    setCurrentMindmapGraphData,
    setCurrentPodcastAudioUrl,
    setCurrentPodcastAudioLoading,
    setCurrentPodcastError,
    setCurrentFlashcards,
    setCurrentTests,
    setUploadedSourceFiles,
    documents,
    addDocument,
    clearDocuments,
    removeDocument,
  } = useDocuments();
  const [isDragging, setIsDragging] = useState(false);
  const navigate = useNavigate();

  const persistedState = readSessionState(HOME_PAGE_STORAGE_KEY, {
    urlInput: "",
    showUrlInput: false,
    messages: [] as Message[],
    input: "",
  });
  const [urlInput, setUrlInput] = useState(persistedState.urlInput);
  const [showUrlInput, setShowUrlInput] = useState(
    persistedState.showUrlInput,
  );

  const [messages, setMessages] = useState<Message[]>(
    persistedState.messages,
  );
  const [input, setInput] = useState(persistedState.input);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // useEffect(() => {
  //   messagesEndRef.current?.scrollIntoView({
  //     behavior: "smooth",
  //   });
  // }, [messages]);

  useEffect(() => {
    writeSessionState(HOME_PAGE_STORAGE_KEY, {
      urlInput,
      showUrlInput,
      messages,
      input,
    });
  }, [input, messages, showUrlInput, urlInput]);

  const processFiles = async (files: File[]) => {
    if (files.length === 0) return;

    setIsLoading(true);
    setCurrentPodcastAudioLoading(true);
    setUploadedSourceFiles(files);
    const formData = new FormData();

    // For RAG, append all files; read text content for persistence
    const newDocuments = [];
    for (const file of files) {
      formData.append("files", file);
      let textContent = "";
      try {
        textContent = await file.text();
      } catch {
        // binary files — leave content empty
      }
      newDocuments.push({
        id: Math.random().toString(36).substr(2, 9),
        name: file.name,
        type: file.type,
        size: file.size,
        content: textContent,
        uploadedAt: new Date(),
      });
    }

    setCurrentFlashcards([]);
    setCurrentTests([]);

    try {
      // Each parallel request gets its OWN FormData to avoid body-consumption issues
      const makeFD = () => {
        const fd = new FormData();
        fd.append("file", files[0]);
        return fd;
      };

      // RAG ingestion uses all files
      const ragResponsePromise = axiosInstance.post(
        API_ROUTES.RAG.INGEST,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } },
      ).catch((err) => { console.error("RAG ingest failed:", err); return null; });

      // Content generation — each with its own FormData and individual error handling
      const mindmapResponsePromise = files.length > 0
        ? axiosInstance.post(API_ROUTES.MINDMAP.FILE, makeFD(), {
            headers: { "Content-Type": "multipart/form-data" },
          }).catch((err) => { console.error("Mindmap generation failed:", err); return null; })
        : Promise.resolve(null);

      const podcastResponsePromise = files.length > 0
        ? axiosInstance.post(API_ROUTES.PODCAST.FILE, makeFD(), {
            headers: { "Content-Type": "multipart/form-data" },
          }).catch((err) => { console.error("Podcast generation failed:", err); return null; })
        : Promise.resolve(null);

      const flashcardsResponsePromise = files.length > 0
        ? axiosInstance.post(API_ROUTES.FLASHCARDS.FILE, makeFD(), {
            headers: { "Content-Type": "multipart/form-data" },
          }).catch((err) => { console.error("Flashcards generation failed:", err); return null; })
        : Promise.resolve(null);

      const [ragResponse, mindmapResponseRaw, podcastResponseRaw, flashcardsResponseRaw] = await Promise.all([
        ragResponsePromise,
        mindmapResponsePromise,
        podcastResponsePromise,
        flashcardsResponsePromise,
      ]);

      // Add documents to context regardless of generation results
      if (ragResponse) {
        console.log("Ingestion successful:", ragResponse.data);
      }
      newDocuments.forEach((doc) => addDocument(doc));

      if (mindmapResponseRaw) {
        console.log("Mindmap generation successful:", mindmapResponseRaw.data);
        const { nodes, edges } = mindmapResponseRaw.data;

        const mapRelevanceToLevel = (relevance: number | undefined): number => {
          if (relevance === undefined) return 0;
          if (relevance >= 0.7) return 0;
          if (relevance >= 0.3 && relevance < 0.7) return 1;
          return 2;
        };

        setCurrentMindmapGraphData({
          nodes: nodes.map((n: any) => ({
            id: String(n.id),
            label: n.label,
            level: mapRelevanceToLevel(n.relevance),
            title: n.title,
            value: n.value,
            relevance: n.relevance,
            summary: n.summary,
          })),
          links: edges.map((e: any) => ({
            source: String(e.from),
            target: String(e.to),
          })),
        });
      }

      const podcastResponse = podcastResponseRaw?.data || null;
      if (podcastResponse && podcastResponse.audio_url) {
        console.log("Podcast generation successful:", podcastResponse);
        setCurrentPodcastAudioUrl(podcastResponse.audio_url);
        setCurrentPodcastError(null);
      } else if (podcastResponse && !podcastResponse.has_audio) {
        const errMsg = podcastResponse.audio_error || "Не удалось сгенерировать аудио";
        console.error("Podcast audio generation failed:", errMsg);
        setCurrentPodcastError(errMsg);
      } else if (!podcastResponseRaw) {
        setCurrentPodcastError("Запрос генерации подкаста не выполнен");
      }

      const flashcardsResponse = flashcardsResponseRaw?.data || null;
      if (flashcardsResponse) {
        setCurrentFlashcards(
          (flashcardsResponse.flashcards || []).map((card: any, index: number) => ({
            id: card.id || `flashcard-${index}`,
            question: card.question,
            answer: card.answer,
          })),
        );
        setCurrentTests(
          (flashcardsResponse.tests || []).map((test: any, index: number) => ({
            id: test.id || `test-${index}`,
            question: test.question,
            options: test.options,
            correctAnswer: test.correct_index ?? test.correctAnswer,
            explanation: test.explanation,
          })),
        );
      }

    } catch (error) {
      console.error("Error during file upload:", error);
    } finally {
      setIsLoading(false);
      setCurrentPodcastAudioLoading(false);
    }
  };

  const handleFileUpload = async (
    files: FileList | null
  ) => {
    if (!files || files.length === 0) return;
    await processFiles(Array.from(files));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileUpload(e.dataTransfer.files);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      Math.round((bytes / Math.pow(k, i)) * 100) / 100 +
      " " +
      sizes[i]
    );
  };

  const handleUrlAdd = async () => {
    if (!urlInput.trim()) return;

    try {
      setIsLoading(true);

      // Add https:// if no protocol is specified
      let url = urlInput.trim();
      if (!url.match(/^https?:\/\//i)) {
        url = "https://" + url;
      }

      // Validate URL
      const urlObj = new URL(url);
      const urlName = urlObj.hostname + urlObj.pathname;

      // Parse URL and ingest into vector DB in one call
      const ingestResponse = await axiosInstance.post(API_ROUTES.PARSER.INGEST, {
        url,
      });

      const { title, indexed, inserted_chunks } = ingestResponse.data;

      if (!indexed || inserted_chunks === 0) {
        throw new Error("Парсер не смог извлечь текст со страницы.");
      }

      // Also parse to get text for content generation (mindmap, podcast, flashcards)
      const parseResponse = await axiosInstance.post(API_ROUTES.PARSER.PARSE, {
        url,
      });

      const parsedText = parseResponse.data?.text || "";

      if (!parsedText.trim()) {
        throw new Error("Парсер вернул пустой текст.");
      }

      const safeFileName =
        `${urlObj.hostname}${urlObj.pathname}`
          .replace(/[^a-zA-Z0-9-_./]+/g, "_")
          .replace(/[/.]+$/g, "") || "parsed-url";

      const docName = title || safeFileName;

      // Add document to local context (store parsed text so other pages can use it)
      addDocument({
        id: Math.random().toString(36).substr(2, 9),
        name: docName,
        type: "text/plain",
        size: parsedText.length,
        content: parsedText,
        uploadedAt: new Date(),
      });

      // Create blob from parsed text for content generation (mindmap, podcast, flashcards)
      // NOTE: use Blob instead of File constructor for Safari compatibility
      const parsedBlob = new Blob([parsedText], { type: "text/plain" });
      const parsedFileName = `${safeFileName}.txt`;

      // Generate content (mindmap, podcast, flashcards) without re-ingesting into RAG
      // Each request needs its OWN FormData — reusing one FormData across
      // multiple parallel requests can cause empty bodies after the first read.
      const makeFD = () => {
        const fd = new FormData();
        fd.append("file", parsedBlob, parsedFileName);
        return fd;
      };

      setCurrentFlashcards([]);
      setCurrentTests([]);
      setCurrentPodcastAudioLoading(true);

      const [mindmapRes, podcastRes, flashcardsRes] = await Promise.all([
        axiosInstance.post(API_ROUTES.MINDMAP.FILE, makeFD(), {
          headers: { "Content-Type": "multipart/form-data" },
        }).catch((err) => { console.error("Mindmap generation failed:", err); return null; }),
        axiosInstance.post(API_ROUTES.PODCAST.FILE, makeFD(), {
          headers: { "Content-Type": "multipart/form-data" },
        }).catch((err) => { console.error("Podcast generation failed:", err); return null; }),
        axiosInstance.post(API_ROUTES.FLASHCARDS.FILE, makeFD(), {
          headers: { "Content-Type": "multipart/form-data" },
        }).catch((err) => { console.error("Flashcards generation failed:", err); return null; }),
      ]);

      setCurrentPodcastAudioLoading(false);

      if (mindmapRes) {
        const { nodes, edges } = mindmapRes.data;
        const mapRelevanceToLevel = (relevance: number | undefined): number => {
          if (relevance === undefined) return 0;
          if (relevance >= 0.7) return 0;
          if (relevance >= 0.3) return 1;
          return 2;
        };
        setCurrentMindmapGraphData({
          nodes: nodes.map((n: any) => ({
            id: String(n.id),
            label: n.label,
            level: mapRelevanceToLevel(n.relevance),
            title: n.title,
            value: n.value,
            relevance: n.relevance,
            summary: n.summary,
          })),
          links: edges.map((e: any) => ({
            source: String(e.from),
            target: String(e.to),
          })),
        });
      }

      if (podcastRes?.data?.audio_url) {
        setCurrentPodcastAudioUrl(podcastRes.data.audio_url);
        setCurrentPodcastError(null);
      } else if (podcastRes?.data && !podcastRes.data.has_audio) {
        setCurrentPodcastError(podcastRes.data.audio_error || "Не удалось сгенерировать аудио");
      } else if (!podcastRes) {
        setCurrentPodcastError("Запрос генерации подкаста не выполнен");
      }

      if (flashcardsRes?.data) {
        setCurrentFlashcards(
          (flashcardsRes.data.flashcards || []).map((card: any, i: number) => ({
            id: card.id || `flashcard-${i}`,
            question: card.question,
            answer: card.answer,
          })),
        );
        setCurrentTests(
          (flashcardsRes.data.tests || []).map((test: any, i: number) => ({
            id: test.id || `test-${i}`,
            question: test.question,
            options: test.options,
            correctAnswer: test.correct_index ?? test.correctAnswer,
            explanation: test.explanation,
          })),
        );
      }

      setUrlInput("");
      setShowUrlInput(false);
    } catch (error) {
      console.error("Error during URL parsing/upload:", error);
      if (axios.isAxiosError(error)) {
        console.error("Parser response data:", error.response?.data);
      }
      const backendMessage =
        axios.isAxiosError(error) &&
        typeof error.response?.data === "string"
          ? error.response.data
          : axios.isAxiosError(error) &&
              typeof error.response?.data?.detail === "string"
            ? error.response.data.detail
            : null;

      alert(
        backendMessage ||
          "Не удалось распарсить URL и загрузить его как документ. Проверьте ссылку и попробуйте снова.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || documents.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await axiosInstance.post(API_ROUTES.RAG.ASK, {
        query: input,
        collection: "docs_ci", // Пока используем значение по умолчанию, можно сделать настраиваемым
      });
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.data.answer,
        sources: Array.isArray(response.data.results)
          ? buildMessageSources(response.data.results)
          : [],
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error during RAG ask:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Извините, произошла ошибка при получении ответа.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockResponse = (
    question: string,
    docs: any[],
  ) => {
    const responses = [
      `На основе анализа загруженных документов (${docs.length} файлов), я могу ответить следующее:\n\n**Основные положения:**\n- Документы содержат важную информацию по теме "${question}"\n- Найдено ${Math.floor(Math.random() * 10) + 3} релевантных упоминаний\n- Контекст указывает на связь с ключевыми концепциями\n\n**Детализация:**\nИсходя из содержания документов, особенно "${docs[0]?.name}", можно выделить несколько важных аспектов. Данные показывают взаимосвязь между различными элементами системы.\n\n**Рекомендации:**\nДля более глубокого понимания рекомендую обратить внимание на разделы, связанные с практическим применением.`,

      `Проанализировав ваш вопрос в контексте документов **"${docs[0]?.name}"** и других источников:\n\n### Ключевые находки\n1. Прямое упоминание темы обнаружено в ${Math.floor(Math.random() * 5) + 1} документах\n2. Связанные концепции найдены в контексте\n3. Дополнительные материалы предоставляют расширенный контекст\n\n### Выводы\nОтвет на ваш вопрос можно найти в нескольких разделах загруженных материалов. Особое внимание стоит уделить практическим примерам и методологии.`,
    ];

    return responses[
      Math.floor(Math.random() * responses.length)
    ];
  };

  const quickActions = [
    {
      name: "Аудиопересказ",
      caption: "Послушайте ИИ-подкаст по вашим источникам",
      path: "/audio",
      color: "#14b8a6",
      gradient: "from-[#14b8a6] to-[#0d9488]",
      icon: Mic,
    },
    {
      name: "Отчеты",
      caption: "Соберите краткое саммари в официальном стиле",
      path: "/reports",
      color: "#8b5cf6",
      gradient: "from-[#8b5cf6] to-[#7c3aed]",
      icon: FileTextIcon,
    },
    {
      name: "Интеллект-карта",
      caption: "Визуализируйте связи между ключевыми понятиями",
      path: "/mindmap",
      color: "#10b981",
      gradient: "from-[#10b981] to-[#059669]",
      icon: Network,
    },
    {
      name: "Карточки и тесты",
      caption: "Сгенерируйте флеш-карточки для самопроверки",
      path: "/flashcards",
      color: "#38C571",
      gradient: "from-[#38C571] to-[#70D116]",
      icon: BookOpen,
    },
    {
      name: "Таблица данных",
      caption: "Структурируйте информацию из текста",
      path: "/data-table",
      color: "#3b82f6",
      gradient: "from-[#3b82f6] to-[#2563eb]",
      icon: Table,
    },
    {
      name: "Видеопересказ",
      caption: "Создайте видео по материалам документов",
      path: "/video",
      color: "#84cc16",
      gradient: "from-[#84cc16] to-[#65a30d]",
      icon: Video,
    },
    {
      name: "Инфографика",
      caption: "Представьте данные в виде цифр и графиков",
      path: "/infographics",
      color: "#f97316",
      gradient: "from-[#f97316] to-[#ea580c]",
      icon: BarChart3,
    },
    {
      name: "Презентация",
      caption:
        "Соберите профессиональные слайды по ключевым темам",
      path: "/presentation",
      color: "#22c55e",
      gradient: "from-[#22c55e] to-[#16a34a]",
      icon: Presentation,
    },
  ];

  const hasClearableOutput =
    documents.length > 0 ||
    messages.length > 0 ||
    Boolean(currentPodcastAudioLoading) ||
    Boolean(currentPodcastAudioUrl) ||
    Boolean(currentMindmapGraphData) ||
    currentFlashcards.length > 0 ||
    currentTests.length > 0;

  const handleClearPageOutput = () => {
    setMessages([]);
    clearDocuments();
    setUploadedSourceFiles([]);
    setCurrentMindmapGraphData(null);
    setCurrentPodcastAudioUrl(null);
    setCurrentPodcastAudioLoading(false);
    setCurrentFlashcards([]);
    setCurrentTests([]);
    writeSessionState(HOME_PAGE_STORAGE_KEY, {
      urlInput,
      showUrlInput,
      messages: [],
      input,
    });
  };

  return (
    <div className="p-6 lg:p-12 max-w-7xl mx-auto relative">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || isLoading}
      />
      {/* Header */}
      <div className="mb-12 text-center">
        <h1 className="text-6xl font-bold mb-4 text-white tracking-tight bg-clip-text">
          KolpakBook
        </h1>
        <p className="text-xl text-gray-400 font-light max-w-2xl mx-auto">
          Интеллектуальная обработка документов для исследований
        </p>
      </div>

      {/* Upload Area */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        className={`
          relative border-2 border-dashed rounded p-12 text-center transition-all
          ${
            isDragging
              ? "border-[#22c55e] bg-[#22c55e]/5"
              : "border-[#262626] bg-[#151515] hover:border-[#404040]"
          }
        `}
      >
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-tr from-[#38C571] to-[#70D116] flex items-center justify-center shadow-lg shadow-[#38C571]/30">
          <Upload className="w-8 h-8 text-white" />
        </div>
        <h3 className="text-xl font-medium text-white mb-3">
          Загрузите документы или добавьте ссылки
        </h3>
        <p className="text-gray-400 mb-6">
          Перетащите файлы сюда или нажмите для выбора
        </p>
        <p className="text-sm text-gray-500 mb-6">
          После загрузки на главной автоматически подготовятся аудио, интеллект-карта, карточки и тесты.
        </p>

        <div className="flex items-center justify-center gap-3 flex-wrap">
          <label className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-tr from-[#38C571] to-[#70D116] text-[#0a0a0a] rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 cursor-pointer transition-all font-medium">
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Загрузка...
              </>
            ) : (
              <>
                <File className="w-4 h-4" />
                Выбрать файлы
              </>
            )}
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.doc"
              onChange={(e) => handleFileUpload(e.target.files)}
              className="hidden"
              disabled={isLoading}
            />
          </label>

          <button
            onClick={() => setShowUrlInput(!showUrlInput)}
            className="inline-flex items-center gap-3 px-6 py-3 bg-[#1a1a1a] border border-[#262626] text-white rounded-full hover:border-[#22c55e] transition-all font-medium"
          >
            <Globe className="w-4 h-4" />
            Добавить URL
          </button>
        </div>

        {/* URL Input */}
        {showUrlInput && (
          <div className="mt-6 max-w-2xl mx-auto">
            <div className="flex gap-3">
              <input
                type="url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyPress={(e) =>
                  e.key === "Enter" && handleUrlAdd()
                }
                placeholder="https://example.com/article"
                className="flex-1 px-5 py-3 bg-[#1a1a1a] border border-[#262626] rounded focus:outline-none focus:border-[#22c55e] transition-colors text-white placeholder:text-gray-600 text-sm"
                disabled={isLoading}
              />
              <button
                onClick={handleUrlAdd}
                disabled={isLoading || !urlInput.trim()}
                className="px-6 py-3 bg-gradient-to-tr from-[#38C571] to-[#70D116] text-[#0a0a0a] rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 transition-all font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Парсим...
                  </span>
                ) : (
                  "Добавить"
                )}
              </button>
            </div>
          </div>
        )}

        <p className="text-xs text-gray-600 mt-4">
          Поддерживаются: PDF, DOCX, TXT, DOC, веб-ссылки
        </p>
      </div>

      {/* Uploaded Documents */}
      {documents.length > 0 && (
        <div className="mt-12">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-white">
              Загруженные источники
            </h2>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[#151515] border border-[#262626] rounded-full">
              <div className="w-1.5 h-1.5 rounded-full bg-[#22c55e]"></div>
              <span className="text-xs font-medium text-gray-400">
                {documents.length}
              </span>
            </div>
          </div>

          <div className="grid gap-3">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="group flex items-center justify-between p-4 bg-[#151515] border border-[#262626] rounded hover:border-[#404040] transition-colors"
              >
                <div className="flex items-center gap-4 flex-1 min-w-0">
                  <div className="w-10 h-10 rounded bg-[#1a1a1a] border border-[#262626] flex items-center justify-center flex-shrink-0">
                    {doc.type === "text/url" ? (
                      <Globe className="w-5 h-5 text-[#14b8a6]" />
                    ) : (
                      <FileText className="w-5 h-5 text-[#22c55e]" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-white truncate text-sm">
                      {doc.name}
                    </h3>
                    {doc.type !== "text/url" && (
                      <p className="text-xs text-gray-500">
                        {formatFileSize(doc.size)} •{" "}
                        {new Date(
                          doc.uploadedAt,
                        ).toLocaleString("ru-RU")}
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => removeDocument(doc.id)}
                  className="p-2 text-gray-600 hover:text-red-500 hover:bg-[#1a1a1a] rounded transition-all opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chat Section - Only show when documents are uploaded */}
      {documents.length > 0 && (
        <div className="mt-12">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Чат
          </h2>

          {/* Chat Messages */}
          <div className="bg-[#151515] border border-[#262626] rounded mb-4">
            <div className="p-6 max-h-[500px] overflow-y-auto space-y-4">
              {messages.length === 0 && (
                <div className="text-center py-8">
                  <div className="w-12 h-12 mx-auto mb-6 rounded-full bg-[#1a1a1a] border border-[#262626] flex items-center justify-center">
                    <FileText className="w-6 h-6 text-[#22c55e]" />
                  </div>
                  <h3 className="text-lg font-medium text-white mb-3">
                    Начните диалог с вашими документами
                  </h3>
                  <p className="text-gray-400 mb-8 text-sm">
                    Примеры вопросов:
                  </p>
                  <div className="grid gap-3 max-w-2xl mx-auto">
                    {[
                      "Какие основные темы освещаются в документах?",
                      "Суммируй ключевые выводы из материалов",
                      "Какие данные и статистика упоминаются?",
                    ].map((example, idx) => (
                      <button
                        key={idx}
                        onClick={() => setInput(example)}
                        className="p-4 bg-[#1a1a1a] border border-[#262626] rounded hover:border-[#404040] transition-colors text-left"
                      >
                        <p className="text-sm text-gray-400">
                          {example}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-3xl rounded px-5 py-4 ${
                      message.role === "user"
                        ? "bg-[#22c55e] text-[#0a0a0a]"
                        : "bg-[#1a1a1a] border border-[#262626] text-white"
                    }`}
                  >
                    <div
                      className={
                        message.role === "user"
                          ? "text-[#0a0a0a]"
                          : "text-gray-300"
                      }
                    >
                      <ReactMarkdown
                        components={{
                          h3: ({ node, ...props }) => (
                            <h3
                              className="font-semibold text-base mb-2 mt-3"
                              {...props}
                            />
                          ),
                          h2: ({ node, ...props }) => (
                            <h2
                              className="font-semibold text-lg mb-3 mt-3"
                              {...props}
                            />
                          ),
                          strong: ({ node, ...props }) => (
                            <strong
                              className="font-semibold"
                              {...props}
                            />
                          ),
                          ul: ({ node, ...props }) => (
                            <ul
                              className="list-disc ml-4 space-y-1 my-2"
                              {...props}
                            />
                          ),
                          ol: ({ node, ...props }) => (
                            <ol
                              className="list-decimal ml-4 space-y-1 my-2"
                              {...props}
                            />
                          ),
                          p: ({ node, ...props }) => (
                            <p
                              className="mb-2 leading-relaxed text-sm"
                              {...props}
                            />
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>

                    {message.sources && (
                      <div className="mt-4 pt-4 border-t border-[#262626]">
                        <p className="text-xs text-gray-500 mb-2 font-medium">
                          Источники:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {message.sources.map(
                            (source, idx) => (
                              <span
                                key={idx}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-[#151515] border border-[#262626] rounded text-xs text-gray-400"
                              >
                                <FileText className="w-3 h-3" />
                                {formatSourceReference(source)}
                              </span>
                            ),
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="max-w-3xl rounded px-5 py-4 bg-[#1a1a1a] border border-[#262626]">
                    <div className="flex items-center gap-3">
                      <Loader2 className="w-4 h-4 animate-spin text-[#22c55e]" />
                      <span className="text-sm text-gray-400">
                        Генерирую ответ...
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Chat Input */}
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) =>
                e.key === "Enter" && handleSend()
              }
              placeholder="Задайте вопрос по документам..."
              className="flex-1 px-5 py-3 bg-[#151515] border border-[#262626] rounded focus:outline-none focus:border-[#22c55e] transition-colors text-white placeholder:text-gray-600 text-sm"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-gradient-to-tr from-[#38C571] to-[#70D116] text-[#0a0a0a] rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all disabled:hover:shadow-none flex items-center gap-2 font-medium"
            >
              <Send className="w-4 h-4" />
              Отправить
            </button>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      {documents.length > 0 && (
        <div className="mt-12">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Доступные действия
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {quickActions.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.path}
                  onClick={() => navigate(action.path)}
                  style={{
                    background: `linear-gradient(135deg, ${action.color} 0%, ${action.color}dd 100%)`,
                  }}
                  className="group relative p-10 rounded transition-all hover:scale-[1.02] text-left overflow-hidden shadow-lg"
                >
                  {/* Icon with reduced opacity */}
                  <div className="absolute top-6 right-6 opacity-10">
                    <Icon
                      className="w-20 h-20 text-white"
                      strokeWidth={1.5}
                    />
                  </div>

                  {/* Glow effect on hover */}
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity blur-2xl"
                    style={{ background: action.color }}
                  ></div>

                  <div className="relative z-10">
                    <Icon
                      className="w-10 h-10 mb-6 text-white drop-shadow-lg"
                      strokeWidth={1.5}
                    />
                    <h3 className="font-semibold text-white text-xl tracking-tight">
                      {action.name}
                    </h3>
                    <p className="text-sm text-white/70 leading-relaxed mt-2">
                      {action.caption}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Features Info */}
      {documents.length === 0 && (
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-[#151515] border border-[#262626] rounded">
            <div className="w-12 h-12 bg-gradient-to-tr from-[#38C571] to-[#70D116] rounded flex items-center justify-center mb-6 shadow-lg shadow-[#38C571]/30">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-base font-medium text-white mb-3">
              RAG-система
            </h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              Загружайте документы и задавайте вопросы по их
              содержанию с помощью AI
            </p>
          </div>

          <div className="p-6 bg-[#151515] border border-[#262626] rounded">
            <div className="w-12 h-12 bg-gradient-to-tr from-[#38C571] to-[#70D116] rounded flex items-center justify-center mb-6 shadow-lg shadow-[#38C571]/30">
              <Database className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-base font-medium text-white mb-3">
              Множество форматов
            </h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              Преобразуйте документы в аудио, видео, тесты,
              интеллект-карты и многое другое
            </p>
          </div>

          <div className="p-6 bg-[#151515] border border-[#262626] rounded">
            <div className="w-12 h-12 bg-gradient-to-tr from-[#38C571] to-[#70D116] rounded flex items-center justify-center mb-6 shadow-lg shadow-[#38C571]/30">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-base font-medium text-white mb-3">
              Конфиденциальность
            </h3>
            <p className="text-sm text-gray-400 leading-relaxed">
              Возможность развертывания в закрытом контуре с
              локальными моделями
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

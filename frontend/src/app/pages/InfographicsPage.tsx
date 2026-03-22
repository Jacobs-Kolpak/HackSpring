import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  Download,
  Loader2,
  TrendingUp,
  Zap,
} from "lucide-react";
import axios from "axios";
import {
  Bar,
  Doughnut,
  Line,
} from "react-chartjs-2";
import {
  ArcElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  BarElement,
  type ChartData,
  type ChartOptions,
} from "chart.js";
import axiosInstance from "../utils/axiosInstance";
import { API_ROUTES } from "../constants/api";
import { useDocuments } from "../context/DocumentContext";
import PageClearButton from "../components/PageClearButton";
import {
  readSessionState,
  writeSessionState,
} from "../utils/sessionState";

ChartJS.register(
  ArcElement,
  BarElement,
  CategoryScale,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
);

interface MetricItem {
  id: string;
  name: string;
  value: string;
  unit: string;
  description: string;
}

interface ChartPoint {
  id: string;
  name: string;
  value: number;
}

interface ApiChartBlock {
  title: string;
  x_axis_label?: string;
  y_axis_label?: string;
  legend: string;
  data: ChartPoint[];
}

interface InfographicsResponse {
  query: string;
  title: string;
  subtitle: string;
  metrics: MetricItem[];
  bar_chart: ApiChartBlock;
  pie_chart: {
    title: string;
    legend: string;
    data: ChartPoint[];
  };
  line_chart: ApiChartBlock;
  key_insights: string[];
  used_chunks: number;
  models_used: string[];
  failed_models: string[];
  download?: {
    mime_type?: string;
    strategy?: string;
    hint?: string;
  };
}

const DEFAULT_QUERY =
  "Построй профессиональную инфографику по ключевым данным, числам, распределениям, трендам и главным выводам из загруженных документов.";

const INFOGRAPHICS_PAGE_STORAGE_KEY =
  "infographics_page_state_v1";

const chartTextColor = "#d4d4d8";
const chartGridColor = "rgba(255,255,255,0.08)";
const piePalette = [
  "#f97316",
  "#fb923c",
  "#fdba74",
  "#22c55e",
  "#14b8a6",
  "#3b82f6",
  "#8b5cf6",
  "#ec4899",
];

const createLineGradient = (
  chart: ChartJS<"line">,
  start: string,
  end: string,
) => {
  const { ctx, chartArea } = chart;

  if (!chartArea) {
    return end;
  }

  const gradient = ctx.createLinearGradient(
    0,
    chartArea.top,
    0,
    chartArea.bottom,
  );
  gradient.addColorStop(0, start);
  gradient.addColorStop(1, end);
  return gradient;
};

const createBarGradient = (
  chart: ChartJS<"bar">,
  start: string,
  end: string,
) => {
  const { ctx, chartArea } = chart;

  if (!chartArea) {
    return end;
  }

  const gradient = ctx.createLinearGradient(
    0,
    chartArea.top,
    0,
    chartArea.bottom,
  );
  gradient.addColorStop(0, start);
  gradient.addColorStop(1, end);
  return gradient;
};

const commonChartOptions = {
  maintainAspectRatio: false,
  responsive: true,
  animation: {
    duration: 700,
    easing: "easeOutQuart" as const,
  },
  plugins: {
    legend: {
      labels: {
        color: chartTextColor,
        boxWidth: 10,
        boxHeight: 10,
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: "#111111",
      borderColor: "rgba(255,255,255,0.12)",
      borderWidth: 1,
      titleColor: "#ffffff",
      bodyColor: "#e4e4e7",
      padding: 12,
      displayColors: true,
    },
  },
} satisfies Partial<ChartOptions<"bar" | "line" | "doughnut">>;

export default function InfographicsPage() {
  const { documents } = useDocuments();
  const persistedState = readSessionState(
    INFOGRAPHICS_PAGE_STORAGE_KEY,
    {
      query: DEFAULT_QUERY,
      selectedSourceName: "all",
      maxTopics: 6,
      errorMessage: null as string | null,
      generatedInfographic: null as InfographicsResponse | null,
    },
  );
  const [query, setQuery] = useState(persistedState.query);
  const [selectedSourceName, setSelectedSourceName] =
    useState<string>(persistedState.selectedSourceName);
  const [maxTopics, setMaxTopics] = useState(
    persistedState.maxTopics,
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(
    persistedState.errorMessage,
  );
  const [generatedInfographic, setGeneratedInfographic] =
    useState<InfographicsResponse | null>(
      persistedState.generatedInfographic,
    );

  useEffect(() => {
    if (
      selectedSourceName !== "all" &&
      !documents.some((doc) => doc.name === selectedSourceName)
    ) {
      setSelectedSourceName("all");
    }
  }, [documents, selectedSourceName]);

  useEffect(() => {
    writeSessionState(INFOGRAPHICS_PAGE_STORAGE_KEY, {
      query,
      selectedSourceName,
      maxTopics,
      errorMessage,
      generatedInfographic,
    });
  }, [
    errorMessage,
    generatedInfographic,
    maxTopics,
    query,
    selectedSourceName,
  ]);

  const barChartRef =
    useRef<ChartJS<"bar", number[], string> | null>(null);
  const doughnutChartRef =
    useRef<ChartJS<"doughnut", number[], string> | null>(null);
  const lineChartRef =
    useRef<ChartJS<"line", number[], string> | null>(null);

  const barChartData = useMemo<ChartData<"bar"> | null>(() => {
    if (!generatedInfographic?.bar_chart?.data?.length) {
      return null;
    }

    return {
      labels: generatedInfographic.bar_chart.data.map(
        (item) => item.name,
      ),
      datasets: [
        {
          label:
            generatedInfographic.bar_chart.legend ||
            generatedInfographic.bar_chart.title,
          data: generatedInfographic.bar_chart.data.map(
            (item) => item.value,
          ),
          borderRadius: 12,
          borderSkipped: false,
          backgroundColor: (context) =>
            createBarGradient(
              context.chart as ChartJS<"bar">,
              "rgba(249, 115, 22, 0.95)",
              "rgba(234, 88, 12, 0.45)",
            ),
          hoverBackgroundColor: "rgba(251, 146, 60, 0.95)",
        },
      ],
    };
  }, [generatedInfographic]);

  const doughnutChartData =
    useMemo<ChartData<"doughnut"> | null>(() => {
      if (!generatedInfographic?.pie_chart?.data?.length) {
        return null;
      }

      return {
        labels: generatedInfographic.pie_chart.data.map(
          (item) => item.name,
        ),
        datasets: [
          {
            label:
              generatedInfographic.pie_chart.legend ||
              generatedInfographic.pie_chart.title,
            data: generatedInfographic.pie_chart.data.map(
              (item) => item.value,
            ),
            borderWidth: 0,
            hoverOffset: 10,
            backgroundColor:
              generatedInfographic.pie_chart.data.map(
                (_, index) =>
                  piePalette[index % piePalette.length],
              ),
          },
        ],
      };
    }, [generatedInfographic]);

  const lineChartData = useMemo<ChartData<"line"> | null>(() => {
    if (!generatedInfographic?.line_chart?.data?.length) {
      return null;
    }

    return {
      labels: generatedInfographic.line_chart.data.map(
        (item) => item.name,
      ),
      datasets: [
        {
          label:
            generatedInfographic.line_chart.legend ||
            generatedInfographic.line_chart.title,
          data: generatedInfographic.line_chart.data.map(
            (item) => item.value,
          ),
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#ffffff",
          pointBorderWidth: 2,
          pointBorderColor: "#8b5cf6",
          tension: 0.35,
          fill: true,
          borderWidth: 3,
          borderColor: "#8b5cf6",
          backgroundColor: (context) =>
            createLineGradient(
              context.chart as ChartJS<"line">,
              "rgba(139, 92, 246, 0.42)",
              "rgba(139, 92, 246, 0.03)",
            ),
        },
      ],
    };
  }, [generatedInfographic]);

  const barChartOptions = useMemo<ChartOptions<"bar">>(
    () => ({
      ...commonChartOptions,
      scales: {
        x: {
          ticks: { color: chartTextColor },
          grid: { display: false },
          title: {
            display: Boolean(
              generatedInfographic?.bar_chart?.x_axis_label,
            ),
            text:
              generatedInfographic?.bar_chart?.x_axis_label ||
              "",
            color: chartTextColor,
          },
        },
        y: {
          beginAtZero: true,
          ticks: { color: chartTextColor },
          grid: { color: chartGridColor },
          title: {
            display: Boolean(
              generatedInfographic?.bar_chart?.y_axis_label,
            ),
            text:
              generatedInfographic?.bar_chart?.y_axis_label ||
              "",
            color: chartTextColor,
          },
        },
      },
    }),
    [generatedInfographic],
  );

  const doughnutChartOptions =
    useMemo<ChartOptions<"doughnut">>(
      () => ({
        ...commonChartOptions,
        cutout: "58%",
      }),
      [],
    );

  const lineChartOptions = useMemo<ChartOptions<"line">>(
    () => ({
      ...commonChartOptions,
      scales: {
        x: {
          ticks: { color: chartTextColor },
          grid: { display: false },
          title: {
            display: Boolean(
              generatedInfographic?.line_chart?.x_axis_label,
            ),
            text:
              generatedInfographic?.line_chart?.x_axis_label ||
              "",
            color: chartTextColor,
          },
        },
        y: {
          beginAtZero: true,
          ticks: { color: chartTextColor },
          grid: { color: chartGridColor },
          title: {
            display: Boolean(
              generatedInfographic?.line_chart?.y_axis_label,
            ),
            text:
              generatedInfographic?.line_chart?.y_axis_label ||
              "",
            color: chartTextColor,
          },
        },
      },
    }),
    [generatedInfographic],
  );

  const handleGenerate = async () => {
    if (!query.trim()) {
      setErrorMessage(
        "Введите запрос: что именно должно показать инфографика.",
      );
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    setGeneratedInfographic(null);

    const payload: Record<string, unknown> = {
      query: query.trim(),
      collection: "docs_ci",
      top_k: 8,
      fetch_k: 20,
      min_score: 0.1,
      dense_weight: 0.7,
      max_topics: maxTopics,
    };

    if (selectedSourceName !== "all") {
      payload.source_name = selectedSourceName;
    }

    try {
      const response = await axiosInstance.post<InfographicsResponse>(
        API_ROUTES.INFOGRAPHICS.CREATE,
        payload,
      );
      setGeneratedInfographic(response.data);
    } catch (error) {
      let message =
        "Не удалось создать инфографику. Проверьте параметры и попробуйте снова.";

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

  const handleChartDownload = (
    chart:
      | ChartJS<"bar", number[], string>
      | ChartJS<"line", number[], string>
      | ChartJS<"doughnut", number[], string>
      | null,
    filename: string,
  ) => {
    if (!chart) {
      return;
    }

    const url = chart.toBase64Image("image/png", 1);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const hasClearableOutput =
    Boolean(generatedInfographic) || Boolean(errorMessage);

  const handleClearPageOutput = () => {
    setGeneratedInfographic(null);
    setErrorMessage(null);
    writeSessionState(INFOGRAPHICS_PAGE_STORAGE_KEY, {
      query,
      selectedSourceName,
      maxTopics,
      errorMessage: null,
      generatedInfographic: null,
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
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#f97316] flex items-center justify-center shadow-2xl shadow-[#f97316]/20">
            <BarChart3 className="w-10 h-10 text-[#f97316]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы для создания инфографики
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
      <div className="max-w-7xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#f97316] rounded-full mb-6">
            <Zap className="w-4 h-4 text-[#f97316]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
              AI Infographics
            </span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Инфографика
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl">
            Chart.js-визуализация данных из документов: метрики,
            распределения, тренды и ключевые выводы.
          </p>
        </div>

        {errorMessage && (
          <div className="mb-6 rounded border border-[#ef4444]/40 bg-[#2a1013] px-5 py-4 text-sm text-red-200">
            {errorMessage}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Запрос для инфографики
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={5}
                  placeholder="Опишите, какую инфографику нужно построить"
                  className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white placeholder:text-gray-600 resize-none focus:outline-none focus:border-[#f97316]"
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Источник
                  </label>
                  <select
                    value={selectedSourceName}
                    onChange={(e) =>
                      setSelectedSourceName(e.target.value)
                    }
                    className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#f97316]"
                  >
                    <option value="all">
                      Все загруженные документы
                    </option>
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.name}>
                        {doc.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Максимум тем
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={maxTopics}
                    onChange={(e) =>
                      setMaxTopics(Number(e.target.value) || 1)
                    }
                    className="w-full px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#f97316]"
                  />
                </div>
              </div>

              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full inline-flex items-center justify-center gap-3 px-6 py-4 bg-gradient-to-r from-[#f97316] to-[#ea580c] text-white rounded-full hover:shadow-2xl hover:shadow-[#f97316]/30 disabled:opacity-60 disabled:cursor-not-allowed transition-all font-semibold"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Генерируем инфографику...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    Создать инфографику
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            {generatedInfographic ? (
              <div className="space-y-6">
                <div className="rounded border border-[#f97316]/20 bg-[radial-gradient(circle_at_top_right,_rgba(249,115,22,0.18),_transparent_42%),linear-gradient(180deg,#141414_0%,#101010_100%)] p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.25em] text-[#fb923c] mb-3">
                        Generated Brief
                      </p>
                      <h2 className="text-3xl font-semibold text-white mb-3">
                        {generatedInfographic.title}
                      </h2>
                      <p className="text-gray-300 leading-7">
                        {generatedInfographic.subtitle}
                      </p>
                    </div>
                    <div className="rounded-full border border-[#f97316]/30 bg-[#f97316]/10 px-3 py-2 text-xs text-orange-200">
                      {generatedInfographic.used_chunks} chunks
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="rounded border border-[#262626] bg-[#101010] p-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                      Models Used
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {generatedInfographic.models_used.length > 0 ? (
                        generatedInfographic.models_used.map(
                          (model) => (
                            <span
                              key={model}
                              className="rounded-full border border-[#262626] bg-[#181818] px-3 py-1 text-xs text-gray-300"
                            >
                              {model}
                            </span>
                          ),
                        )
                      ) : (
                        <span className="text-sm text-gray-500">
                          Не указано
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="rounded border border-[#262626] bg-[#101010] p-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                      Download Strategy
                    </div>
                    <div className="text-sm text-gray-300">
                      {generatedInfographic.download?.strategy ||
                        "Не указано"}
                    </div>
                    {generatedInfographic.download?.hint && (
                      <div className="text-xs text-gray-500 mt-2">
                        {generatedInfographic.download.hint}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full min-h-[420px] rounded border border-dashed border-[#262626] bg-[#0f0f0f] flex items-center justify-center p-8 text-center">
                <div>
                  <div className="w-16 h-16 mx-auto mb-5 rounded-full bg-[#f97316]/10 border border-[#f97316]/20 flex items-center justify-center">
                    <BarChart3 className="w-8 h-8 text-[#f97316]" />
                  </div>
                  <p className="text-white font-medium mb-2">
                    Инфографика пока не построена
                  </p>
                  <p className="text-sm text-gray-400 max-w-sm">
                    Задайте запрос слева, настройте источник и
                    получите Chart.js-визуализацию по документам.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {generatedInfographic && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5 mt-8 mb-8">
              {generatedInfographic.metrics.map((metric) => (
                <div
                  key={metric.id}
                  className="rounded border border-[#262626] bg-[#151515] p-5 relative overflow-hidden"
                >
                  <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#f97316]/70 to-transparent" />
                  <div className="flex items-start justify-between gap-3 mb-4">
                    <span className="text-sm text-gray-400">
                      {metric.name}
                    </span>
                    <TrendingUp className="w-4 h-4 text-[#f97316]" />
                  </div>
                  <div className="text-3xl font-semibold text-white mb-2">
                    {metric.value}
                    {metric.unit ? (
                      <span className="text-lg text-gray-400 ml-2">
                        {metric.unit}
                      </span>
                    ) : null}
                  </div>
                  <p className="text-sm text-gray-500 leading-6">
                    {metric.description}
                  </p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="rounded border border-[#262626] bg-[#151515] p-6">
                <div className="flex items-center justify-between gap-4 mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white">
                      {generatedInfographic.bar_chart.title}
                    </h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Сравнение ключевых показателей.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      handleChartDownload(
                        barChartRef.current,
                        "infographic-bar-chart.png",
                      )
                    }
                    className="inline-flex items-center gap-2 rounded-full border border-[#262626] px-4 py-2 text-sm text-gray-300 transition hover:border-[#f97316]"
                  >
                    <Download className="w-4 h-4" />
                    PNG
                  </button>
                </div>
                <div className="h-[320px]">
                  {barChartData ? (
                    <Bar
                      ref={barChartRef}
                      data={barChartData}
                      options={barChartOptions}
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-gray-500">
                      Данные для столбчатого графика отсутствуют.
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded border border-[#262626] bg-[#151515] p-6">
                <div className="flex items-center justify-between gap-4 mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white">
                      {generatedInfographic.pie_chart.title}
                    </h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Распределение по сегментам.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      handleChartDownload(
                        doughnutChartRef.current,
                        "infographic-pie-chart.png",
                      )
                    }
                    className="inline-flex items-center gap-2 rounded-full border border-[#262626] px-4 py-2 text-sm text-gray-300 transition hover:border-[#f97316]"
                  >
                    <Download className="w-4 h-4" />
                    PNG
                  </button>
                </div>
                <div className="h-[320px]">
                  {doughnutChartData ? (
                    <Doughnut
                      ref={doughnutChartRef}
                      data={doughnutChartData}
                      options={doughnutChartOptions}
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-gray-500">
                      Данные для круговой диаграммы отсутствуют.
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded border border-[#262626] bg-[#151515] p-6 lg:col-span-2">
                <div className="flex items-center justify-between gap-4 mb-6">
                  <div>
                    <h3 className="text-lg font-semibold text-white">
                      {generatedInfographic.line_chart.title}
                    </h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Динамика и развитие по шкале времени или этапов.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      handleChartDownload(
                        lineChartRef.current,
                        "infographic-line-chart.png",
                      )
                    }
                    className="inline-flex items-center gap-2 rounded-full border border-[#262626] px-4 py-2 text-sm text-gray-300 transition hover:border-[#f97316]"
                  >
                    <Download className="w-4 h-4" />
                    PNG
                  </button>
                </div>
                <div className="h-[340px]">
                  {lineChartData ? (
                    <Line
                      ref={lineChartRef}
                      data={lineChartData}
                      options={lineChartOptions}
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-gray-500">
                      Данные для линейного графика отсутствуют.
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-6 mt-6">
              <div className="rounded border border-[#262626] bg-[#151515] p-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Ключевые выводы
                </h3>
                <div className="space-y-3">
                  {generatedInfographic.key_insights.length > 0 ? (
                    generatedInfographic.key_insights.map(
                      (insight, index) => (
                        <div
                          key={`${insight}-${index}`}
                          className="flex gap-3 rounded border border-[#262626] bg-[#101010] p-4"
                        >
                          <div className="mt-0.5 w-7 h-7 shrink-0 rounded-full bg-[#f97316]/15 border border-[#f97316]/30 flex items-center justify-center text-xs text-[#fb923c]">
                            {index + 1}
                          </div>
                          <p className="text-sm leading-6 text-gray-300">
                            {insight}
                          </p>
                        </div>
                      ),
                    )
                  ) : (
                    <div className="text-sm text-gray-500">
                      Ключевые инсайты не были возвращены.
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded border border-[#262626] bg-[#151515] p-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Диагностика ответа
                </h3>

                <div className="space-y-4">
                  <div className="rounded border border-[#262626] bg-[#101010] p-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                      Исходный запрос
                    </div>
                    <p className="text-sm text-gray-300 leading-6">
                      {generatedInfographic.query}
                    </p>
                  </div>

                  <div className="rounded border border-[#262626] bg-[#101010] p-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                      Неуспешные модели
                    </div>
                    {generatedInfographic.failed_models.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {generatedInfographic.failed_models.map(
                          (model) => (
                            <span
                              key={model}
                              className="rounded-full border border-[#ef4444]/30 bg-[#2a1013] px-3 py-1 text-xs text-red-200"
                            >
                              {model}
                            </span>
                          ),
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm text-[#22c55e]">
                        <AlertCircle className="w-4 h-4" />
                        Ошибок по моделям не зафиксировано
                      </div>
                    )}
                  </div>

                  <div className="rounded border border-[#262626] bg-[#101010] p-4">
                    <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">
                      Использованный источник
                    </div>
                    <p className="text-sm text-gray-300">
                      {selectedSourceName === "all"
                        ? "Все загруженные документы"
                        : selectedSourceName}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

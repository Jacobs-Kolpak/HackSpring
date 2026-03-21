import { useEffect, useState } from 'react';
import {
  Table as TableIcon,
  Download,
  Zap,
  Loader2,
  FileText
} from 'lucide-react';
import axiosInstance from '../utils/axiosInstance';
import { API_ROUTES } from '../constants/api';
import { useDocuments } from '../context/DocumentContext';
import {
  readSessionState,
  writeSessionState,
} from '../utils/sessionState';
import PageClearButton from '../components/PageClearButton';

interface TableData {
  title?: string;
  headers: string[];
  rows: string[][];
}

const DATA_TABLE_PAGE_STORAGE_KEY = 'data_table_page_state_v1';

const normalizeTableResponse = (payload: unknown): TableData => {
  const parsed =
    typeof payload === 'string' ? JSON.parse(payload) : payload;

  if (
    parsed &&
    typeof parsed === 'object' &&
    'columns' in parsed &&
    'rows' in parsed
  ) {
    const table = parsed as {
      title?: unknown;
      columns: unknown[];
      rows: unknown[][];
    };
    return {
      title:
        typeof table.title === 'string' ? table.title : undefined,
      headers: table.columns.map((item) => String(item)),
      rows: table.rows.map((row) => row.map((item) => String(item ?? ''))),
    };
  }

  if (
    parsed &&
    typeof parsed === 'object' &&
    'headers' in parsed &&
    'rows' in parsed
  ) {
    const table = parsed as {
      title?: unknown;
      headers: unknown[];
      rows: unknown[][];
    };
    return {
      title:
        typeof table.title === 'string' ? table.title : undefined,
      headers: table.headers.map((item) => String(item)),
      rows: table.rows.map((row) => row.map((item) => String(item ?? ''))),
    };
  }

  throw new Error('Неизвестный формат табличных данных.');
};

export default function DataTablePage() {
  const { documents, uploadedSourceFiles } = useDocuments();
  const persistedState = readSessionState(
    DATA_TABLE_PAGE_STORAGE_KEY,
    {
      tableData: null as TableData | null,
      hint: '',
    },
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [tableData, setTableData] = useState<TableData | null>(
    persistedState.tableData,
  );
  const [hint, setHint] = useState(persistedState.hint);
  const [format] = useState('json');

  useEffect(() => {
    writeSessionState(DATA_TABLE_PAGE_STORAGE_KEY, {
      tableData,
      hint,
    });
  }, [hint, tableData]);

  const latestUploadedFile =
    uploadedSourceFiles.length > 0
      ? uploadedSourceFiles[uploadedSourceFiles.length - 1]
      : null;

  const handleGenerate = async () => {
    if (!hint.trim()) {
      alert('Введите запрос для построения таблицы.');
      return;
    }

    if (!latestUploadedFile) {
      alert('На главной странице должен быть загружен хотя бы один файл.');
      return;
    }

    setIsGenerating(true);
    setTableData(null);

    try {
      const formData = new FormData();
      formData.append('file', latestUploadedFile);
      formData.append('hint', hint);
      formData.append('format', format);

      console.log('Table file request:', {
        fileName: latestUploadedFile.name,
        hint,
        format,
      });

      const response = await axiosInstance.post(
        API_ROUTES.TABLE.FILE,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        },
      );

      console.log('Table response:', response.data);
      setTableData(normalizeTableResponse(response.data));
    } catch (error) {
      console.error('Error generating table:', error);
      alert('Не удалось построить таблицу. Проверьте запрос и попробуйте снова.');
    } finally {
      setIsGenerating(false);
    }
  };

  const exportCSV = () => {
    if (!tableData) return;

    const csvContent = [
      tableData.headers.join(','),
      ...tableData.rows.map((row) =>
        row
          .map((cell) => `"${cell.replaceAll('"', '""')}"`)
          .join(','),
      ),
    ].join('\n');

    const blob = new Blob([csvContent], {
      type: 'text/csv;charset=utf-8;',
    });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'data_export.csv';
    link.click();
  };

  const hasClearableOutput = Boolean(tableData);

  const handleClearPageOutput = () => {
    setTableData(null);
    writeSessionState(DATA_TABLE_PAGE_STORAGE_KEY, {
      tableData: null,
      hint,
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
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#3b82f6] flex items-center justify-center shadow-2xl shadow-[#3b82f6]/20">
            <TableIcon className="w-10 h-10 text-[#3b82f6]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы на главной странице для извлечения структурированных данных
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
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#3b82f6] rounded-full mb-6">
            <Zap className="w-4 h-4 text-[#3b82f6]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">AI Data Extraction</span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Таблица данных
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl">
            Таблица строится по последнему файлу, который вы загрузили на главной странице. От вас нужен только запрос.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="bg-[#151515] border border-[#262626] rounded p-8">
            <div className="mb-6 rounded-lg border border-[#262626] bg-[#0f0f0f] p-5">
              <div className="flex items-center gap-3 mb-3">
                <FileText className="w-5 h-5 text-[#3b82f6]" />
                <h2 className="text-lg font-semibold text-white">
                  Исходный файл
                </h2>
              </div>
              <p className="text-sm text-gray-400">
                Для построения таблицы будет использован:
              </p>
              <p className="mt-2 text-white font-medium">
                {latestUploadedFile?.name || 'Файл не найден'}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Запрос пользователя
              </label>
              <textarea
                value={hint}
                onChange={(e) => setHint(e.target.value)}
                placeholder="Например: Составить таблицу с ключевыми понятиями"
                className="w-full min-h-[180px] px-4 py-3 bg-[#0f0f0f] border border-[#262626] rounded text-white focus:outline-none focus:border-[#3b82f6]"
              />
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating || !latestUploadedFile}
              className="mt-8 w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-[#3b82f6] to-[#2563eb] text-white rounded-full hover:shadow-2xl hover:shadow-[#3b82f6]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Строим таблицу...
                </>
              ) : (
                <>
                  <TableIcon className="w-5 h-5" />
                  Построить таблицу
                </>
              )}
            </button>
          </div>

          <div className="bg-[#151515] border border-[#262626] rounded overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#262626] bg-[#0f0f0f]">
              <h2 className="text-xl font-semibold text-white">
                Результат
              </h2>
              {tableData && (
                <button
                  onClick={exportCSV}
                  className="px-4 py-2 bg-[#151515] border border-[#262626] rounded-full hover:border-[#3b82f6] transition-all flex items-center gap-2 text-gray-300 font-medium"
                >
                  <Download className="w-4 h-4" />
                  Скачать CSV
                </button>
              )}
            </div>

            {tableData ? (
              <>
                {tableData.title && (
                  <div className="px-6 py-4 border-b border-[#262626] bg-[#101010]">
                    <p className="text-sm text-gray-400 mb-1">Название таблицы</p>
                    <h3 className="text-lg font-semibold text-white">
                      {tableData.title}
                    </h3>
                  </div>
                )}
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-[#0a0a0a]">
                      <tr>
                        {tableData.headers.map((header, idx) => (
                          <th
                            key={idx}
                            className="px-6 py-4 text-left text-sm font-bold text-white border-b border-[#262626]"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.rows.map((row, rowIdx) => (
                        <tr
                          key={rowIdx}
                          className={rowIdx % 2 === 0 ? 'bg-[#151515]' : 'bg-[#0a0a0a]'}
                        >
                          {row.map((cell, cellIdx) => (
                            <td
                              key={cellIdx}
                              className="px-6 py-4 text-sm text-gray-300 border-b border-[#262626]"
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="min-h-[420px] flex flex-col items-center justify-center text-center p-8">
                <TableIcon className="w-12 h-12 text-[#3b82f6] mb-4" />
                <p className="text-white text-lg font-medium mb-2">
                  Таблица еще не построена
                </p>
                <p className="text-gray-400 max-w-md">
                  Введите запрос слева и нажмите кнопку построения. Таблица будет создана по последнему файлу с главной страницы.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

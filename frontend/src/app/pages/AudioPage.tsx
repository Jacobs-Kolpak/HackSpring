import { useEffect, useState } from 'react';
import { useDocuments } from '../context/DocumentContext';
import { useNavigate } from 'react-router-dom';
import { Download, House, Loader2, Volume2, AlertCircle, Radio, Lightbulb } from 'lucide-react';
import { API_BASE_URL } from '../constants/api';
import PageClearButton from '../components/PageClearButton';

export default function AudioPage() {
  const {
    documents,
    currentPodcastAudioUrl,
    currentPodcastAudioLoading,
    setCurrentPodcastAudioLoading,
    setCurrentPodcastAudioUrl,
    currentPodcastError,
    setCurrentPodcastError,
  } = useDocuments();
  const navigate = useNavigate();

  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [downloadFilename, setDownloadFilename] = useState('podcast.wav');
  const [isPlayerLoading, setIsPlayerLoading] = useState(false);
  const [audioLoadError, setAudioLoadError] = useState(false);

  const resolveAudioRequestUrl = (url: string) => {
    if (/^https?:\/\//i.test(url)) {
      return url;
    }
    try {
      const apiOrigin = new URL(API_BASE_URL, window.location.origin).origin;
      return new URL(url, apiOrigin).toString();
    } catch {
      return url;
    }
  };

  useEffect(() => {
    if (!currentPodcastAudioUrl) {
      setAudioUrl(null);
      setDownloadFilename('podcast.wav');
      setIsPlayerLoading(false);
      setAudioLoadError(false);
      return;
    }

    setAudioUrl(resolveAudioRequestUrl(currentPodcastAudioUrl));
    setDownloadFilename(
      currentPodcastAudioUrl.split('/').pop()?.split('?')[0] ||
        'podcast.wav',
    );
    setIsPlayerLoading(true);
    setAudioLoadError(false);
  }, [currentPodcastAudioUrl]);

  const handleDownload = () => {
    if (!audioUrl) return;

    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = downloadFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isAudioBusy = currentPodcastAudioLoading || isPlayerLoading;
  const hasClearableOutput =
    Boolean(audioUrl) || Boolean(currentPodcastAudioUrl) || audioLoadError || Boolean(currentPodcastError);

  const handleClearPageOutput = () => {
    setAudioUrl(null);
    setDownloadFilename('podcast.wav');
    setIsPlayerLoading(false);
    setAudioLoadError(false);
    setCurrentPodcastAudioUrl(null);
    setCurrentPodcastAudioLoading(false);
    setCurrentPodcastError(null);
  };

  if (documents.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6 bg-[#0a0a0a]">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-xl">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#10b981] flex items-center justify-center shadow-2xl shadow-[#10b981]/20">
            <Volume2 className="w-10 h-10 text-[#10b981]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документ на главной странице. После этого аудиопересказ будет сформирован автоматически.
          </p>
          <button
            onClick={() => navigate('/')}
            className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#10b981] to-[#059669] rounded-full text-[#0a0a0a] font-semibold"
          >
            <House className="w-4 h-4" />
            На главную
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-300 p-6 lg:p-12">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput || currentPodcastAudioLoading}
      />
      <div className="max-w-5xl mx-auto">
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#14b8a6] rounded-full mb-6">
            <Radio className="w-4 h-4 text-[#14b8a6]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
              AI Audio Generation
            </span>
          </div>
          <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
            Аудиопересказ
          </h1>
          <p className="text-lg text-gray-400">
            Генерация диалога двух AI-ведущих, обсуждающих
            материал
          </p>
        </div>

        <div className="bg-[#1a1a1a] border border-[#262626] rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-[#10b981]/15 border border-[#10b981]/30 flex items-center justify-center">
              <Volume2 className="w-6 h-6 text-[#10b981]" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Блок воспроизведения</h3>
              <p className="text-sm text-gray-400">Готовое аудио по последнему загруженному документу</p>
            </div>
          </div>

          {audioUrl && !audioLoadError ? (
            <>
              <div className="rounded-xl border border-[#262626] bg-[#0f0f0f] p-4">
                <div className="flex items-center justify-between gap-4 mb-3">
                  <p className="text-sm text-gray-400">
                    Файл: <span className="text-white">{downloadFilename}</span>
                  </p>
                  {isAudioBusy && (
                    <span className="inline-flex items-center gap-2 text-xs text-[#10b981]">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {currentPodcastAudioLoading
                        ? 'Формируем аудио...'
                        : 'Буферизуем аудио...'}
                    </span>
                  )}
                </div>

                <audio
                  controls
                  preload="metadata"
                  src={audioUrl}
                  className="w-full"
                  onLoadStart={() => {
                    setIsPlayerLoading(true);
                    setAudioLoadError(false);
                  }}
                  onLoadedMetadata={() => setIsPlayerLoading(false)}
                  onCanPlay={() => setIsPlayerLoading(false)}
                  onWaiting={() => setIsPlayerLoading(true)}
                  onPlaying={() => setIsPlayerLoading(false)}
                  onError={() => {
                    setAudioLoadError(true);
                    setIsPlayerLoading(false);
                  }}
                >
                  Ваш браузер не поддерживает элемент аудио.
                </audio>
              </div>

              <button
                onClick={handleDownload}
                className="mt-4 w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-[#10b981] text-[#0a0a0a] rounded-full font-semibold hover:bg-[#0d9a71] transition-colors"
              >
                <Download className="w-5 h-5" />
                Скачать аудио
              </button>
            </>
          ) : (
            <div className="rounded-xl border border-dashed border-[#262626] bg-[#0f0f0f] p-6 min-h-[220px] flex flex-col items-center justify-center text-center">
              {audioLoadError || currentPodcastError ? (
                <AlertCircle className="w-10 h-10 text-[#f97316] mb-4" />
              ) : isAudioBusy ? (
                <Loader2 className="w-10 h-10 text-[#10b981] mb-4 animate-spin" />
              ) : (
                <Volume2 className="w-10 h-10 text-[#10b981] mb-4" />
              )}

              <p className="text-white font-medium mb-2">
                {audioLoadError
                  ? 'Не удалось открыть аудиофайл'
                  : currentPodcastError
                    ? 'Ошибка генерации аудио'
                    : isAudioBusy
                      ? 'Аудио загружается'
                      : 'Аудио еще не готово'}
              </p>

              <p className="text-sm text-gray-400 max-w-sm">
                {audioLoadError
                  ? 'Сервер оборвал соединение при выдаче большого WAV-файла. Попробуйте открыть страницу еще раз или скачать файл напрямую позже.'
                  : currentPodcastError
                    ? currentPodcastError
                    : currentPodcastAudioLoading
                      ? 'Система формирует аудиопересказ по загруженному документу.'
                      : isPlayerLoading
                        ? 'Браузер получает поток аудио и подготавливает его к воспроизведению. Это может занять время для больших файлов.'
                        : 'Вернитесь на главную страницу и загрузите документ. После обработки аудиопересказ появится здесь автоматически.'}
              </p>

              <button
                onClick={() => navigate('/')}
                className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#10b981] to-[#059669] rounded-full text-[#0a0a0a] font-semibold disabled:opacity-70 disabled:cursor-not-allowed"
                disabled={isAudioBusy}
              >
                {isAudioBusy ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {currentPodcastAudioLoading ? 'Формируем аудио...' : 'Загружаем аудио...'}
                  </>
                ) : (
                  <>
                    <House className="w-4 h-4" />
                    Перейти на главную
                  </>
                )}
              </button>
            </div>
          )}
          <div className="mt-8 p-6 bg-[#151515] border border-[#14b8a6]/30 rounded">
          <p className="text-sm text-gray-300 leading-relaxed flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-[#14b8a6] flex-shrink-0 mt-0.5" />
            <span>
              <strong className="text-white">Совет:</strong>{" "}
              Аудиопересказ создает диалог между двумя
              AI-ведущими, которые обсуждают ключевые моменты из
              ваших документов в выбранном стиле. Идеально для
              прослушивания в дороге или во время тренировки.
            </span>
          </p>
        </div>
        </div>
      </div>
    </div>
  );
}

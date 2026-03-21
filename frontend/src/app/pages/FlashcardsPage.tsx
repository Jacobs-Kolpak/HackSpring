import {
  BookOpen,
  RotateCw,
  Check,
  X,
  Zap,
  ChevronRight,
  FileText,
  House,
} from "lucide-react";
import { useDocuments } from "../context/DocumentContext";
import { useState } from "react";
import { useNavigate } from "react-router";
import PageClearButton from "../components/PageClearButton";

export default function FlashcardsPage() {
  const {
    documents,
    currentFlashcards,
    currentTests,
    setCurrentFlashcards,
    setCurrentTests,
  } = useDocuments();
  const [mode, setMode] = useState<
    "select" | "flashcards" | "test"
  >("select");
  const [currentCard, setCurrentCard] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [testAnswers, setTestAnswers] = useState<
    Record<string, number>
  >({});
  const [showResults, setShowResults] = useState(false);
  const navigate = useNavigate();

  const handleNextCard = () => {
    if (currentCard < currentFlashcards.length - 1) {
      setCurrentCard(currentCard + 1);
      setShowAnswer(false);
    }
  };

  const handlePrevCard = () => {
    if (currentCard > 0) {
      setCurrentCard(currentCard - 1);
      setShowAnswer(false);
    }
  };

  const handleTestAnswer = (
    questionId: string,
    answerIndex: number,
  ) => {
    setTestAnswers((prev) => ({
      ...prev,
      [questionId]: answerIndex,
    }));
  };

  const submitTest = () => {
    setShowResults(true);
  };

  const calculateScore = () => {
    let correct = 0;
    currentTests.forEach((q) => {
      if (testAnswers[q.id] === q.correctAnswer) {
        correct++;
      }
    });
    return { correct, total: currentTests.length };
  };

  const hasClearableOutput =
    currentFlashcards.length > 0 || currentTests.length > 0;

  const resetLocalState = () => {
    setMode("select");
    setCurrentCard(0);
    setShowAnswer(false);
    setTestAnswers({});
    setShowResults(false);
  };

  const handleClearPageOutput = () => {
    setCurrentFlashcards([]);
    setCurrentTests([]);
    resetLocalState();
  };

  if (documents.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-md">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#38C571] flex items-center justify-center shadow-2xl shadow-[#38C571]/20">
            <BookOpen className="w-10 h-10 text-[#38C571]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы на главной странице для генерации карточек и тестов.
          </p>
          <button
            onClick={() => navigate("/")}
            className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#38C571] to-[#70D116] rounded-full text-[#0a0a0a] font-semibold"
          >
            <House className="w-4 h-4" />
            На главную
          </button>
        </div>
      </div>
    );
  }

  if (currentFlashcards.length === 0 && currentTests.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-xl">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#38C571] flex items-center justify-center shadow-2xl shadow-[#38C571]/20">
            <BookOpen className="w-10 h-10 text-[#38C571]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Материалы еще не готовы
          </h2>
          <p className="text-gray-400 text-lg">
            Карточки и тесты формируются при загрузке файла на главной странице. Откройте главную и добавьте документ.
          </p>
          <button
            onClick={() => navigate("/")}
            className="mt-6 inline-flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-[#38C571] to-[#70D116] rounded-full text-[#0a0a0a] font-semibold"
          >
            <House className="w-4 h-4" />
            Перейти на главную
          </button>
        </div>
      </div>
    );
  }

  if (mode === "select") {
    return (
      <div className="min-h-screen p-6 lg:p-12">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="max-w-4xl mx-auto">
          <div className="mb-10">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#151515] border border-[#38C571] rounded-full mb-6">
              <Zap className="w-4 h-4 text-[#38C571]" />
              <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">
                AI Learning Tools
              </span>
            </div>
            <h1 className="text-5xl font-bold mb-3 text-white tracking-tight">
              Карточки и тесты
            </h1>
            <p className="text-lg text-gray-400">
              Материалы уже сформированы из файла, загруженного на главной странице.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-8 bg-[#151515] border-2 border-[#262626] rounded hover:border-[#38C571] transition-all text-left group shadow-lg hover:shadow-2xl hover:shadow-[#38C571]/20">
              <div className="w-16 h-16 bg-gradient-to-br from-[#38C571] to-[#70D116] rounded flex items-center justify-center mb-4 shadow-lg shadow-[#38C571]/30">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">
                Флеш-карточки
              </h3>
              <p className="text-gray-400 mb-4">
                Набор карточек с вопросами и ответами для запоминания материала
              </p>
              <div className="mb-4 text-sm text-gray-300">
                Подготовлено карточек: {currentFlashcards.length}
              </div>
              <button
                onClick={() => {
                  setCurrentCard(0);
                  setShowAnswer(false);
                  setMode("flashcards");
                }}
                className="w-full inline-flex items-center justify-center gap-2 px-6 py-3 bg-[#38C571] text-[#0a0a0a] rounded-full font-semibold hover:bg-[#32b264] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={currentFlashcards.length === 0}
              >
                <BookOpen className="w-5 h-5" />
                Открыть карточки
              </button>
            </div>

            <button
              onClick={() => setMode("test")}
              disabled={currentTests.length === 0}
              className="p-8 bg-[#151515] border-2 border-[#262626] rounded hover:border-[#70D116] transition-all text-left group shadow-lg hover:shadow-2xl hover:shadow-[#70D116]/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="w-16 h-16 bg-gradient-to-br from-[#70D116] to-[#65a30d] rounded flex items-center justify-center mb-4 shadow-lg shadow-[#70D116]/30">
                <Check className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">
                Тест с вариантами
              </h3>
              <p className="text-gray-400 mb-4">
                Интерактивный тест с множественным выбором для проверки знаний
              </p>
              <p className="text-sm text-gray-300 mb-4">
                Подготовлено вопросов: {currentTests.length}
              </p>
              <div className="flex items-center text-[#70D116] font-medium">
                Открыть тест
                <ChevronRight className="w-5 h-5 ml-1" />
              </div>
            </button>
          </div>

          <div className="mt-8 p-6 bg-[#151515] border border-[#38C571]/30 rounded">
            <h3 className="font-medium text-white mb-2 flex items-center gap-2">
              <FileText className="w-4 h-4 text-[#38C571]" />
              Источники для генерации
            </h3>
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="text-sm text-gray-300"
                >
                  • {doc.name}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (mode === "flashcards") {
    return (
      <div className="min-h-screen p-6 lg:p-12">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                Flash-карточки
              </h1>
              <p className="text-gray-400">
                Карточка {currentCard + 1} из{" "}
                {currentFlashcards.length}
              </p>
            </div>
            <button
              onClick={() => setMode("select")}
              className="px-4 py-2 bg-[#151515] border border-[#262626] rounded hover:border-[#38C571] text-gray-400 transition-all"
            >
              Назад
            </button>
          </div>

          <div
            onClick={() => setShowAnswer(!showAnswer)}
            className="relative h-96 cursor-pointer mb-8"
          >
            <div
              className={`absolute inset-0 transition-all duration-500 ${
                showAnswer
                  ? "opacity-0 rotate-y-180"
                  : "opacity-100"
              }`}
            >
              <div className="h-full bg-gradient-to-br from-[#38C571] to-[#70D116] rounded shadow-2xl shadow-[#38C571]/30 p-8 flex flex-col items-center justify-center text-white">
                <BookOpen className="w-12 h-12 mb-6 opacity-80" />
                <h2 className="text-2xl font-bold text-center mb-4">
                  {currentFlashcards[currentCard]?.question}
                </h2>
                <p className="text-white/80 text-sm">
                  Нажмите для просмотра ответа
                </p>
              </div>
            </div>

            <div
              className={`absolute inset-0 transition-all duration-500 ${
                showAnswer
                  ? "opacity-100"
                  : "opacity-0 rotate-y-180"
              }`}
            >
              <div className="h-full bg-[#151515] border border-[#262626] rounded shadow-2xl p-8 flex flex-col items-center justify-center">
                <Check className="w-12 h-12 mb-6 text-[#38C571]" />
                <p className="text-lg text-gray-300 text-center leading-relaxed">
                  {currentFlashcards[currentCard]?.answer}
                </p>
                <p className="text-gray-500 text-sm mt-4">
                  Нажмите для возврата к вопросу
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevCard}
              disabled={currentCard === 0}
              className="px-6 py-3 bg-[#151515] border border-[#262626] rounded hover:border-[#38C571] transition-all disabled:opacity-50 disabled:cursor-not-allowed text-gray-300"
            >
              ← Предыдущая
            </button>

            <button
              onClick={() => {
                setCurrentCard(0);
                setShowAnswer(false);
              }}
              className="p-3 bg-[#151515] border border-[#262626] rounded hover:border-[#38C571] transition-all text-gray-400"
              title="Начать сначала"
            >
              <RotateCw className="w-5 h-5" />
            </button>

            <button
              onClick={handleNextCard}
              disabled={currentCard === currentFlashcards.length - 1}
              className="px-6 py-3 bg-gradient-to-r from-[#38C571] to-[#70D116] rounded shadow-lg shadow-[#38C571]/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium"
            >
              Следующая →
            </button>
          </div>

          <div className="mt-6">
            <div className="flex gap-2">
              {currentFlashcards.map((_, idx) => (
                <div
                  key={idx}
                  className={`flex-1 h-2 rounded-full transition-colors ${
                    idx === currentCard
                      ? "bg-[#38C571] shadow-lg shadow-[#38C571]/50"
                      : idx < currentCard
                        ? "bg-[#70D116]"
                        : "bg-[#262626]"
                  }`}
                />
              ))}
            </div>
          </div>
          <button
            onClick={() => setMode("test")}
            disabled={currentTests.length === 0}
            className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-[#38C571] to-[#70D116] text-white rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
          >
            Перейти к тесту
          </button>
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
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">
              Тест с вариантами ответов
            </h1>
            <p className="text-gray-400">
              Выберите правильные ответы на вопросы
            </p>
          </div>
          <button
            onClick={() => setMode("select")}
            className="px-4 py-2 bg-[#151515] border border-[#262626] rounded hover:border-[#38C571] text-gray-400 transition-all"
          >
            Назад
          </button>
        </div>

        {!showResults ? (
          <>
            <div className="space-y-6">
              {currentTests.map((q, qIdx) => (
                <div
                  key={q.id}
                  className="bg-[#151515] border border-[#262626] rounded p-6"
                >
                  <h3 className="font-bold text-white mb-4">
                    {qIdx + 1}. {q.question}
                  </h3>
                  <div className="space-y-3">
                    {q.options.map((option, oIdx) => (
                      <button
                        key={oIdx}
                        onClick={() =>
                          handleTestAnswer(q.id, oIdx)
                        }
                        className={`w-full p-4 rounded border-2 text-left transition-all ${
                          testAnswers[q.id] === oIdx
                            ? "border-[#38C571] bg-[#38C571]/10 shadow-lg shadow-[#38C571]/20 text-white"
                            : "border-[#262626] hover:border-[#404040] text-gray-300"
                        }`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={submitTest}
              disabled={
                Object.keys(testAnswers).length !==
                currentTests.length
              }
              className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-[#38C571] to-[#70D116] text-white rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
            >
              Завершить тест
            </button>
          </>
        ) : (
          <div className="bg-[#151515] border border-[#262626] rounded p-8 mb-6 shadow-2xl">
            <div className="text-center mb-8">
              <div className="w-24 h-24 bg-gradient-to-br from-[#38C571] to-[#70D116] rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg shadow-[#38C571]/30">
                <span className="text-4xl font-bold text-white">
                  {Math.round(
                    (calculateScore().correct /
                      calculateScore().total) *
                      100,
                  )}
                  %
                </span>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">
                Тест завершен!
              </h2>
              <p className="text-gray-400">
                Правильных ответов:{" "}
                <span className="text-white font-semibold">
                  {calculateScore().correct}
                </span>{" "}
                из {calculateScore().total}
              </p>
            </div>

            <div className="space-y-4">
              {currentTests.map((q, idx) => {
                const isCorrect =
                  testAnswers[q.id] === q.correctAnswer;
                return (
                  <div
                    key={q.id}
                    className={`p-4 rounded border-2 ${
                      isCorrect
                        ? "border-[#38C571] bg-[#38C571]/10"
                        : "border-red-600 bg-red-600/10"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {isCorrect ? (
                        <Check className="w-6 h-6 text-[#38C571] flex-shrink-0 mt-1" />
                      ) : (
                        <X className="w-6 h-6 text-red-500 flex-shrink-0 mt-1" />
                      )}
                      <div className="flex-1">
                        <h4 className="font-medium text-white mb-2">
                          {idx + 1}. {q.question}
                        </h4>
                        <p className="text-sm text-gray-300">
                          <strong>Ваш ответ:</strong>{" "}
                          {q.options[testAnswers[q.id]]}
                        </p>
                        {!isCorrect && (
                          <p className="text-sm text-[#38C571] mt-1">
                            <strong>Правильный ответ:</strong>{" "}
                            {q.options[q.correctAnswer]}
                          </p>
                        )}
                        {q.explanation && (
                          <p className="text-sm text-gray-400 mt-2">
                            <strong>Пояснение:</strong> {q.explanation}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <button
              onClick={() => {
                setTestAnswers({});
                setShowResults(false);
              }}
              className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-[#38C571] to-[#70D116] text-white rounded-full hover:shadow-2xl hover:shadow-[#38C571]/30 transition-all font-medium"
            >
              Пройти заново
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

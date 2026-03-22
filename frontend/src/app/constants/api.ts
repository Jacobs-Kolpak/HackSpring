export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/jacobs';
export const API_ROUTES = {
  AUTH: {
    LOGIN: `${API_BASE_URL}/auth/login`,
    REGISTER: `${API_BASE_URL}/auth/register`,
    ME: `${API_BASE_URL}/auth/me`,
    REFRESH: `${API_BASE_URL}/auth/refresh`,
    LOGOUT: `${API_BASE_URL}/auth/logout`,
  },
  RAG: {
    INGEST: `${API_BASE_URL}/rag/ingest`,
    RETRIEVE: `${API_BASE_URL}/rag/retrieve`,
    ASK: `${API_BASE_URL}/rag/ask`,
  },
  MINDMAP: {
    FILE: `${API_BASE_URL}/mindmap/file`,
  },
  PODCAST: {
    FILE: `${API_BASE_URL}/podcast/file`,
    AUDIO: `${API_BASE_URL}/podcast/audio`,
  },
  PARSER: {
    PARSE: `${API_BASE_URL}/parser/parse`,
    INGEST: `${API_BASE_URL}/parser/ingest`,
  },
  FLASHCARDS: {
    FILE: `${API_BASE_URL}/flashcards/file`,
  },
  SUMMARY: {
    FILE: `${API_BASE_URL}/summary/file`,
  },
  INFOGRAPHICS: {
    CREATE: `${API_BASE_URL}/infographics`,
  },
  PRESENTATION: {
    GENERATE: `${API_BASE_URL}/presentation/generate`,
    GENERATE_WITH_TEMPLATE: `${API_BASE_URL}/presentation/generate-with-template`,
    FROM_RESULTS: `${API_BASE_URL}/presentation/from-results`,
    DOWNLOAD: `${API_BASE_URL}/presentation/download`,
  },
  TABLE: {
    TEXT: `${API_BASE_URL}/table/text`,
    FILE: `${API_BASE_URL}/table/file`,
  },
  VIDEO: {
    GENERATE: `${API_BASE_URL}/video/from-file`,
  },
};

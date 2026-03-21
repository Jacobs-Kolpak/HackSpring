import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

export interface Node {
  id: string;
  label: string;
  level?: number;
  title?: string;
  value?: number;
  relevance?: number;
  summary?: string;
  x?: number;
  y?: number;
}

export interface Edge {
  from: string;
  to: string;
}

export interface MindmapGraphData {
  nodes: Node[];
  links: { source: string; target: string }[];
}

export interface FlashcardStudyItem {
  id: string;
  question: string;
  answer: string;
}

export interface TestQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation?: string;
}

export interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  content: string;
  uploadedAt: Date;
  mindmapGraphData?: MindmapGraphData;
}

interface DocumentContextType {
  documents: Document[];
  addDocument: (doc: Document) => void;
  removeDocument: (id: string) => void;
  clearDocuments: () => void;
  currentMindmapGraphData: MindmapGraphData | null;
  setCurrentMindmapGraphData: (data: MindmapGraphData | null) => void;
  currentPodcastAudioUrl: string | null;
  setCurrentPodcastAudioUrl: (url: string | null) => void;
  currentPodcastAudioLoading: boolean;
  setCurrentPodcastAudioLoading: (loading: boolean) => void;
  currentFlashcards: FlashcardStudyItem[];
  setCurrentFlashcards: (flashcards: FlashcardStudyItem[]) => void;
  currentTests: TestQuestion[];
  setCurrentTests: (tests: TestQuestion[]) => void;
  uploadedSourceFiles: File[];
  setUploadedSourceFiles: (files: File[]) => void;
}

const DocumentContext = createContext<DocumentContextType | undefined>(undefined);

// ---- sessionStorage persistence helpers ----
const STORAGE_KEY = 'docCtx';

interface PersistedState {
  documents: Document[];
  mindmap: MindmapGraphData | null;
  podcastUrl: string | null;
  flashcards: FlashcardStudyItem[];
  tests: TestQuestion[];
}

function loadPersistedState(): PersistedState | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedState;
    // Restore Date objects for uploadedAt
    if (parsed.documents) {
      parsed.documents = parsed.documents.map(d => ({
        ...d,
        uploadedAt: new Date(d.uploadedAt),
      }));
    }
    return parsed;
  } catch {
    return null;
  }
}

function savePersistedState(state: PersistedState) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // sessionStorage full or unavailable — ignore
  }
}

export function DocumentProvider({ children }: { children: ReactNode }) {
  const persisted = loadPersistedState();

  const [documents, setDocuments] = useState<Document[]>(persisted?.documents ?? []);
  const [currentMindmapGraphData, setCurrentMindmapGraphData] = useState<MindmapGraphData | null>(persisted?.mindmap ?? null);
  const [currentPodcastAudioUrl, setCurrentPodcastAudioUrl] = useState<string | null>(persisted?.podcastUrl ?? null);
  const [currentPodcastAudioLoading, setCurrentPodcastAudioLoading] = useState(false);
  const [currentFlashcards, setCurrentFlashcards] = useState<FlashcardStudyItem[]>(persisted?.flashcards ?? []);
  const [currentTests, setCurrentTests] = useState<TestQuestion[]>(persisted?.tests ?? []);
  const [uploadedSourceFiles, setUploadedSourceFiles] = useState<File[]>([]);

  // Persist serialisable state to sessionStorage on every change
  useEffect(() => {
    savePersistedState({
      documents,
      mindmap: currentMindmapGraphData,
      podcastUrl: currentPodcastAudioUrl,
      flashcards: currentFlashcards,
      tests: currentTests,
    });
  }, [documents, currentMindmapGraphData, currentPodcastAudioUrl, currentFlashcards, currentTests]);

  const addDocument = useCallback((doc: Document) => {
    setDocuments(prev => [...prev, doc]);
  }, []);

  const removeDocument = useCallback((id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  }, []);

  const clearDocuments = useCallback(() => {
    setDocuments([]);
    setCurrentMindmapGraphData(null);
    setCurrentPodcastAudioUrl(null);
    setCurrentPodcastAudioLoading(false);
    setCurrentFlashcards([]);
    setCurrentTests([]);
    setUploadedSourceFiles([]);
    sessionStorage.removeItem(STORAGE_KEY);
  }, []);

  return (
    <DocumentContext.Provider value={{
      documents,
      addDocument,
      removeDocument,
      clearDocuments,
      currentMindmapGraphData,
      setCurrentMindmapGraphData,
      currentPodcastAudioUrl,
      setCurrentPodcastAudioUrl,
      currentPodcastAudioLoading,
      setCurrentPodcastAudioLoading,
      currentFlashcards,
      setCurrentFlashcards,
      currentTests,
      setCurrentTests,
      uploadedSourceFiles,
      setUploadedSourceFiles
    }}>
      {children}
    </DocumentContext.Provider>
  );
}

export function useDocuments() {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within DocumentProvider');
  }
  return context;
}

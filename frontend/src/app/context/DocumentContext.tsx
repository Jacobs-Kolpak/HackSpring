import React, { createContext, useContext, useState, ReactNode } from 'react';

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

export function DocumentProvider({ children }: { children: ReactNode }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [currentMindmapGraphData, setCurrentMindmapGraphData] = useState<MindmapGraphData | null>(null);
  const [currentPodcastAudioUrl, setCurrentPodcastAudioUrl] = useState<string | null>(null);
  const [currentPodcastAudioLoading, setCurrentPodcastAudioLoading] = useState(false);
  const [currentFlashcards, setCurrentFlashcards] = useState<FlashcardStudyItem[]>([]);
  const [currentTests, setCurrentTests] = useState<TestQuestion[]>([]);
  const [uploadedSourceFiles, setUploadedSourceFiles] = useState<File[]>([]);

  const addDocument = (doc: Document) => {
    setDocuments(prev => [...prev, doc]);
  };

  const removeDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  };

  const clearDocuments = () => {
    setDocuments([]);
  };

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

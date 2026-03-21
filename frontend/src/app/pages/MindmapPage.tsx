import { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useDocuments } from '../context/DocumentContext';
import { useNavigate } from 'react-router-dom';
import { Network, Zap } from 'lucide-react';
import PageClearButton from '../components/PageClearButton';

interface Node {
  id: string;
  label: string;
  level: number;
  summary?: string;
  x?: number;
  y?: number;
}

export default function MindmapPage() {
  const {
    documents,
    currentMindmapGraphData,
    setCurrentMindmapGraphData,
  } = useDocuments();
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const graphRef = useRef<any>();
  const contentRef = useRef<HTMLDivElement | null>(null);
  const [contentSize, setContentSize] = useState({ width: 1200, height: 700 });

  const getColor = (level?: number) => {
    const actualLevel: number = typeof level === 'number' && !isNaN(level) ? level : 0;
    const colors = ["#27da9eff", "#059669", "#024733ff"];
    return colors[actualLevel % colors.length] || colors[0];
  };

  useEffect(() => {
    if (currentMindmapGraphData) {
      setZoom(1);
      setOffset({ x: 0, y: 0 });
      setSelectedNode(null);
    }
  }, [currentMindmapGraphData]);

  useEffect(() => {
    if (graphRef.current) {
      const linkForce = graphRef.current.d3Force('link') as any;
      const chargeForce = graphRef.current.d3Force('charge') as any;

      if (linkForce) {
        linkForce.strength(0.2);
        linkForce.distance(130);
      }

      if (chargeForce) {
        chargeForce.strength(-230);
      }
    }
  }, [graphRef]);

  useEffect(() => {
    const updateSizes = () => {
      if (contentRef.current) {
        setContentSize({
          width: contentRef.current.clientWidth,
          height: contentRef.current.clientHeight,
        });
      }
    };

    updateSizes();

    const resizeObserver = new ResizeObserver(() => {
      updateSizes();
    });

    if (contentRef.current) {
      resizeObserver.observe(contentRef.current);
    }

    window.addEventListener('resize', updateSizes);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updateSizes);
    };
  }, []);

  const hasClearableOutput =
    Boolean(currentMindmapGraphData) &&
    currentMindmapGraphData.nodes.length > 0;

  const handleClearPageOutput = () => {
    setCurrentMindmapGraphData(null);
    setSelectedNode(null);
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  if (!currentMindmapGraphData || currentMindmapGraphData.nodes.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center p-6">
        <PageClearButton
          onClick={handleClearPageOutput}
          disabled={!hasClearableOutput}
        />
        <div className="text-center max-w-md">
          <div className="w-20 h-20 mx-auto mb-6 rounded bg-[#151515] border-2 border-[#10b981] flex items-center justify-center shadow-2xl shadow-[#10b981]/20">
            <Network className="w-10 h-10 text-[#10b981]" />
          </div>
          <h2 className="text-3xl font-semibold text-white mb-3">
            Нет загруженных документов
          </h2>
          <p className="text-gray-400 text-lg">
            Загрузите документы для создания интеллект-карты
          </p>
        </div>
      </div>
    );
  }

  const graphNodesCount = currentMindmapGraphData.nodes.length;
  const graphEdgesCount = currentMindmapGraphData.links.length;
  const sidePanelWidth = Math.min(
    440,
    Math.max(320, Math.floor(contentSize.width * 0.34)),
  );

  return (
    <div className="h-screen flex flex-col">
      <PageClearButton
        onClick={handleClearPageOutput}
        disabled={!hasClearableOutput}
      />
      {/* Header */}
      <div className="bg-[#0f0f0f] border-b border-[#262626] px-5 lg:px-8 py-4 flex items-center justify-between shrink-0">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#151515] border border-[#10b981] rounded-full mb-2">
            <Zap className="w-4 h-4 text-[#10b981]" />
            <span className="text-sm font-medium text-gray-300 uppercase tracking-wider">AI Mind Mapping</span>
          </div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Интеллект-карта</h1>
          <p className="text-sm text-gray-400 mt-1">
            Визуализация связей между ключевыми понятиями
          </p>
        </div>
      </div>

      {/* ForceGraph2D Canvas */}
      <div ref={contentRef} className="flex-1 min-h-0 bg-[#0a0a0a] flex overflow-hidden">
        <div className="flex-1 relative">
          <ForceGraph2D
            ref={graphRef}
            graphData={currentMindmapGraphData}
            nodeLabel="label"
            nodeAutoColorBy="level"
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = (node as Node).label;
              const level = (node as Node).level;
              const fontSize = 12 / globalScale;

              ctx.beginPath();
              ctx.arc(node.x || 0, node.y || 0, 8, 0, 2 * Math.PI);
              ctx.fillStyle = getColor(level);
              ctx.fill();

              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillStyle = "white";
              ctx.fillText(label, (node.x || 0), (node.y || 0) + 14);
            }}
            linkColor={() => "#9ca3af"}
            linkWidth={0.8}
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={1}
            onNodeDragEnd={(node) => {
              node.fx = node.x;
              node.fy = node.y;
            }}
            onNodeClick={(node) => {
              setSelectedNode(node as Node);
            }}
            height={Math.max(320, contentSize.height)}
            width={Math.max(320, contentSize.width - sidePanelWidth)}
          />

          {/* Legend */}
          <div className="absolute bottom-3 left-3 p-3 bg-[#0f0f0f] rounded-lg shadow-lg border border-[#262626] text-gray-300">
            <h4 className="text-sm font-semibold mb-2 text-white">Уровни</h4>
            <div className="space-y-1 text-xs">
              {[
                { label: "Главная тема", level: 0 },
                { label: "Концепции", level: 1 },
                { label: "Подтемы", level: 2 }
              ].map((item) => (
                <div key={item.level} className="flex items-center">
                  <span className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: getColor(item.level) }}></span>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Instructions */}
          <div className="absolute bottom-3 right-3 p-3 bg-[#0f0f0f] rounded-lg shadow-lg border border-[#262626] text-gray-300 w-56">
            <h4 className="text-sm font-semibold mb-2 text-white">Инструкции</h4>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>Перетаскивайте узлы, чтобы изменить их положение.</li>
              <li>Используйте колесико мыши для приближения/отдаления.</li>
              <li>Нажмите на узел, чтобы открыть описание справа.</li>
            </ul>
          </div>
        </div>

        <aside
          className="shrink-0 border-l border-[#262626] bg-[#0f0f0f] p-5 flex flex-col overflow-y-auto overflow-x-hidden"
          style={{ width: `${sidePanelWidth}px` }}
        >
          <div className="mb-4 shrink-0">
            <h3 className="text-lg font-semibold text-white mb-1.5">
              Информация об узле
            </h3>
            <p className="text-xs text-gray-400">
              Выберите точку на карте, чтобы увидеть подробное описание.
            </p>
          </div>

          {selectedNode ? (
            <div className="rounded-lg border border-[#10b981]/30 bg-[#111111] p-4 min-h-0 overflow-y-auto">
              <div className="mb-3">
                <p className="text-xs uppercase tracking-wider text-[#10b981] mb-2">
                  Выбранный узел
                </p>
                <h4 className="text-xl font-semibold text-white break-words">
                  {selectedNode.label}
                </h4>
              </div>
              <p className="text-sm leading-relaxed text-gray-300 break-words">
                {selectedNode.summary || "Для этого узла описание отсутствует."}
              </p>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-[#262626] bg-[#111111] p-4 text-sm text-gray-400">
              Нажмите на любую зеленую точку на карте, чтобы открыть ее summary в этой панели.
            </div>
          )}
        </aside>
      </div>

      {/* Stats */}
      <div className="bg-[#0f0f0f] border-t border-[#262626] px-5 py-3 shrink-0">
        <div className="flex items-center justify-center gap-6 text-xs text-gray-400">
          <span className="flex items-center gap-2">
            <span className="text-white font-semibold">{graphNodesCount}</span> узлов
          </span>
          <span className="flex items-center gap-2">
            <span className="text-white font-semibold">{graphEdgesCount}</span> связей
          </span>
          <span className="flex items-center gap-2">
            <span className="text-white font-semibold">{documents.length}</span> источников
          </span>
        </div>
      </div>
    </div>
  );
}

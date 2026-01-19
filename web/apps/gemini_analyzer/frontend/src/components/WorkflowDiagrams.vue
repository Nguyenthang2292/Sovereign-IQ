<template>
  <div class="workflow-diagrams max-w-7xl mx-auto px-6 py-6">
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-white mb-2">{{ $t('workflowDiagrams.title') }}</h1>
      <p class="text-gray-400">{{ $t('workflowDiagrams.subtitle') }}</p>
    </div>

    <!-- Analyzer Tabs -->
    <div class="glass-panel rounded-lg p-1 mb-6 flex gap-2" role="tablist">
      <button @click="activeTab = 'voting'" role="tab" :aria-selected="activeTab === 'voting'" :class="[
        'flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-300',
        activeTab === 'voting'
          ? 'btn-gradient text-white shadow-lg'
          : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
      ]">
        {{ $t('workflowDiagrams.votingAnalyzer.title') }}
      </button>
      <button @click="activeTab = 'hybrid'" :class="[
        'flex-1 px-4 py-3 rounded-lg font-medium transition-all duration-300',
        activeTab === 'hybrid'
          ? 'btn-gradient text-white shadow-lg'
          : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
      ]">
        {{ $t('workflowDiagrams.hybridAnalyzer.title') }}
      </button>
    </div>

    <!-- Voting Analyzer Content -->
    <div v-if="activeTab === 'voting'" class="space-y-6">
      <div class="glass-panel rounded-lg p-6">
        <h2 class="text-2xl font-bold text-white mb-4">{{ $t('workflowDiagrams.votingAnalyzer.workflowTitle') }}</h2>
        <p class="text-gray-400 mb-2">
          {{ $t('workflowDiagrams.votingAnalyzer.description') }}
        </p>
        <p class="text-sm text-purple-400 mb-6 flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
          {{ $t('workflowDiagrams.votingAnalyzer.clickHint') }}
        </p>

        <!-- Mermaid Diagram -->
        <div class="mermaid-container bg-gray-900/50 rounded-lg p-4 mb-6">
          <div ref="votingDiagram" class="mermaid-diagram"></div>
        </div>

        <!-- Module Details -->
        <ModuleDetails ref="votingModuleDetails" :modules="votingModules" :highlighted-module="highlightedModule" />
      </div>
    </div>

    <!-- Hybrid Analyzer Content -->
    <div v-if="activeTab === 'hybrid'" class="space-y-6">
      <div class="glass-panel rounded-lg p-6">
        <h2 class="text-2xl font-bold text-white mb-4">{{ $t('workflowDiagrams.hybridAnalyzer.workflowTitle') }}</h2>
        <p class="text-gray-400 mb-2">
          {{ $t('workflowDiagrams.hybridAnalyzer.description') }}
        </p>
        <p class="text-sm text-purple-400 mb-6 flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
          {{ $t('workflowDiagrams.hybridAnalyzer.clickHint') }}
        </p>

        <!-- Mermaid Diagram -->
        <div class="mermaid-container bg-gray-900/50 rounded-lg p-4 mb-6">
          <div ref="hybridDiagram" class="mermaid-diagram"></div>
        </div>

        <!-- Module Details -->
        <ModuleDetails ref="hybridModuleDetails" :modules="hybridModules" :highlighted-module="highlightedModule" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import mermaid from 'mermaid'
import ModuleDetails from './ModuleDetails.vue'

const { t } = useI18n()

// Debug logging - only enabled in development
const DEBUG = import.meta.env.DEV

function debugLog(...args: any[]) {
  if (DEBUG) console.log('[WorkflowDiagrams]', ...args)
}

function debugWarn(...args: any[]) {
  if (DEBUG) console.warn('[WorkflowDiagrams]', ...args)
}

// Interfaces
interface ModuleData {
  name: string
  path: string
  description: string
  inputs: string[]
  outputs: string[]
  keyFiles: string[]
  keyFunctions: string[]
}

type NodeMap = Record<string, string | null>

const activeTab = ref<'voting' | 'hybrid'>('voting')
const votingDiagram = ref<HTMLElement | null>(null)
const hybridDiagram = ref<HTMLElement | null>(null)
const votingModuleDetails = ref<InstanceType<typeof ModuleDetails> | null>(null)
const hybridModuleDetails = ref<InstanceType<typeof ModuleDetails> | null>(null)
const highlightedModule = ref<string | null>(null)

// Store cleanup functions for event listeners
const cleanupFunctions: (() => void)[] = []

// Store timeout IDs for cleanup
const timeoutIds: ReturnType<typeof setTimeout>[] = []

// Mapping diagram node names to module names
// Include multiple variations to handle text extraction differences
const votingNodeToModuleMap: NodeMap = {
  'ATC': 'ATC Analyzer',
  'ATC Scan': 'ATC Analyzer',
  'ATC Signal': 'ATC Analyzer',
  'ATC_Calc': 'ATC Analyzer',
  'Range Oscillator': 'Range Oscillator',
  'OSC_Calc': 'Range Oscillator',
  'SPC 3 Strategies': 'SPC (Simplified Percentile Clustering)',
  'SPC_Calc': 'SPC (Simplified Percentile Clustering)',
  'SPC': 'SPC (Simplified Percentile Clustering)',
  'XGBoost': 'XGBoost',
  'XGB_Calc': 'XGBoost',
  'HMM': 'HMM (Hidden Markov Model)',
  'HMM_Calc': 'HMM (Hidden Markov Model)',
  'Random Forest': 'Random Forest',
  'RF_Calc': 'Random Forest',
  'Decision Matrix Voting System': 'Decision Matrix Voting System',
  'Decision Matrix': 'Decision Matrix Voting System',
  'Voting System': 'Decision Matrix Voting System',
  'Voting': 'Decision Matrix Voting System',
  'Calculate All Indicators in Parallel': null, // Skip this node
  'Final Results': null, // Skip this node
  'Start': null, // Skip this node
  'End': null // Skip this node
}

const hybridNodeToModuleMap: NodeMap = {
  'ATC': 'ATC Analyzer',
  'ATC Scan': 'ATC Analyzer',
  'Range Oscillator Early Filter': 'Range Oscillator (Early Filter)',
  'Range Oscillator': 'Range Oscillator (Early Filter)',
  'OSC_Filter': 'Range Oscillator (Early Filter)',
  'Calculate SPC Signals 3 Strategies': 'SPC (Simplified Percentile Clustering)',
  'Calculate SPC Signals': 'SPC (Simplified Percentile Clustering)',
  'SPC': 'SPC (Simplified Percentile Clustering)',
  'Decision Matrix Voting Optional': 'Decision Matrix Voting (Optional)',
  'Decision Matrix': 'Decision Matrix Voting (Optional)',
  'Decision Matrix Voting': 'Decision Matrix Voting (Optional)',
  'Decision': 'Decision Matrix Voting (Optional)',
  'Fallback to ATC Only': null, // Skip this node
  'Final Results': null, // Skip this node
  'Start': null, // Skip this node
  'End': null // Skip this node
}

// Initialize Mermaid - ensure it's initialized before use
let mermaidInitialized = false

function initializeMermaid() {
  if (mermaidInitialized) return

  try {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        primaryColor: '#8b5cf6',
        primaryTextColor: '#fff',
        primaryBorderColor: '#a78bfa',
        lineColor: '#6b7280',
        secondaryColor: '#4b5563',
        tertiaryColor: '#374151'
      },
      flowchart: {
        useMaxWidth: false,
        htmlLabels: false,
        curve: 'basis',
        nodeSpacing: 100,
        rankSpacing: 60,
        padding: 50,
        // @ts-ignore
        paddingX: 25,
        // @ts-ignore
        paddingY: 20
      },
      securityLevel: 'strict'
    })
    mermaidInitialized = true
    debugLog('Mermaid initialized successfully')
  } catch (error) {
    console.error('Error initializing Mermaid:', error)
  }
}

// Initialize immediately
initializeMermaid()

// Voting Analyzer Mermaid Diagram
const votingDiagramCode = `
flowchart TD
    Start([Start]) --> ATC[ATC Scan]
    ATC --> Parallel[Calculate All Indicators<br/>in Parallel]
    Parallel --> ATC_Calc[ATC Signal]
    Parallel --> OSC_Calc[Range Oscillator]
    Parallel --> SPC_Calc[SPC 3 Strategies]
    Parallel --> XGB_Calc[XGBoost]
    Parallel --> HMM_Calc[HMM]
    Parallel --> RF_Calc[Random Forest]
    ATC_Calc --> Voting[Decision Matrix<br/>Voting System]
    OSC_Calc --> Voting
    SPC_Calc --> Voting
    XGB_Calc --> Voting
    HMM_Calc --> Voting
    RF_Calc --> Voting
    Voting --> Results[Final Results]
    Results --> End([End])
    
    style Start fill:#8b5cf6,stroke:#a78bfa,stroke-width:2px,color:#fff
    style End fill:#8b5cf6,stroke:#a78bfa,stroke-width:2px,color:#fff
    style ATC fill:#4b5563,stroke:#6b7280,stroke-width:2px,color:#fff
    style Parallel fill:#6366f1,stroke:#818cf8,stroke-width:2px,color:#fff
    style Voting fill:#026E00,stroke:#029E00,stroke-width:2px,color:#fff
    style Results fill:#F5320B,stroke:#F5590B,stroke-width:2px,color:#fff
`

// Hybrid Analyzer Mermaid Diagram
const hybridDiagramCode = `
flowchart TD
    Start([Start]) --> ATC[ATC Scan]
    ATC --> OSC_Filter[Range Oscillator<br/>Filter]
    OSC_Filter -->|Pass| SPC[SPC Signals<br/>3 Strategies]
    OSC_Filter -->|Fail| Fallback[Fallback<br/>ATC Only]
    SPC --> Decision[Decision Matrix<br/>Voting]
    Fallback --> Decision
    Decision --> Results[Results]
    Results --> End([End])
    
    style Start fill:#8b5cf6,stroke:#a78bfa,stroke-width:2px,color:#fff
    style End fill:#8b5cf6,stroke:#a78bfa,stroke-width:2px,color:#fff
    style ATC fill:#4b5563,stroke:#6b7280,stroke-width:2px,color:#fff
    style OSC_Filter fill:#6366f1,stroke:#818cf8,stroke-width:2px,color:#fff
    style SPC fill:#10b981,stroke:#34d399,stroke-width:2px,color:#fff
    style Decision fill:#F5320B,stroke:#F5590B,stroke-width:2px,color:#fff
    style Results fill:#ec4899,stroke:#f472b6,stroke-width:2px,color:#fff
    style Fallback fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#fff
`

// Module data for Voting Analyzer
const votingModules: ModuleData[] = [
  {
    name: 'ATC Analyzer',
    path: 'modules/adaptive_trend',
    description: 'Adaptive Trend Classification - Market trend detection using KAMA-based indicators',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Timeframe (m15, m30, h1, h4, d1...)',
      'KAMA parameters (fast_period, slow_period, signal_period)'
    ],
    outputs: [
      'LONG signals (buy signals)',
      'SHORT signals (sell signals)',
      'Trend classification (UPTREND, DOWNTREND, SIDEWAYS)',
      'Signal strength (0.0 - 1.0)'
    ],
    keyFiles: ['modules/adaptive_trend/cli/main.py', 'modules/adaptive_trend/core/analyzer.py'],
    keyFunctions: ['run_auto_scan()', 'classify_trend()', 'calculate_kama()']
  },
  {
    name: 'Range Oscillator',
    path: 'modules/range_oscillator',
    description: 'Overbought/oversold zone detection using price range analysis',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Oscillator length (default: 14)',
      'Multiplier (default: 2.0)',
      'Strategies (breakout, mean_reversion, trend_following)'
    ],
    outputs: [
      'Oscillator signal (1 for LONG, -1 for SHORT)',
      'Confidence score (0.0 - 1.0)',
      'Zone classification (OVERBOUGHT, OVERSOLD, NEUTRAL)'
    ],
    keyFiles: ['modules/range_oscillator/core/oscillator.py', 'core/signal_calculators.py'],
    keyFunctions: ['get_range_oscillator_signal()', 'calculate_oscillator()']
  },
  {
    name: 'SPC (Simplified Percentile Clustering)',
    path: 'modules/simplified_percentile_clustering',
    description: 'Percentile-based clustering with 3 strategies: Cluster Transition, Regime Following, Mean Reversion',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'SPC parameters (k: 2, lookback: 100, p_low: 0.1, p_high: 0.9)',
      'Strategy parameters (min_signal_strength, min_rel_pos_change, etc.)'
    ],
    outputs: [
      'SPC signals from 3 strategies (cluster_transition, regime_following, mean_reversion)',
      'Aggregated SPC vote (LONG, SHORT, NONE)',
      'Signal strength (0.0 - 1.0)'
    ],
    keyFiles: ['modules/simplified_percentile_clustering/core/clustering.py', 'modules/simplified_percentile_clustering/aggregation.py'],
    keyFunctions: ['get_spc_signal()', 'aggregate()', 'SPCVoteAggregator']
  },
  {
    name: 'XGBoost',
    path: 'modules/xgboost',
    description: 'Machine learning prediction using gradient boosting',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Technical indicators (RSI, MACD, EMA, etc.)',
      'Trained model (.pkl or .joblib file)'
    ],
    outputs: [
      'XGBoost signal (LONG, SHORT, NONE)',
      'Confidence score (0.0 - 1.0)'
    ],
    keyFiles: ['core/signal_calculators.py'],
    keyFunctions: ['get_xgboost_signal()']
  },
  {
    name: 'HMM (Hidden Markov Model)',
    path: 'modules/hmm',
    description: 'Hidden Markov Model for regime detection and signal prediction',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'HMM parameters (n_states: 3, n_iter: 100)',
      'Window size (default: 50 candles)'
    ],
    outputs: [
      'HMM signal (LONG, SHORT, NONE)',
      'Confidence score (0.0 - 1.0)',
      'Regime state (BULLISH, BEARISH, NEUTRAL)'
    ],
    keyFiles: ['core/signal_calculators.py'],
    keyFunctions: ['get_hmm_signal()']
  },
  {
    name: 'Random Forest',
    path: 'modules/random_forest',
    description: 'Random Forest ensemble model for signal prediction',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Technical indicators (RSI, MACD, EMA, etc.)',
      'Trained model (.joblib file)'
    ],
    outputs: [
      'Random Forest signal (LONG, SHORT, NONE)',
      'Confidence score (0.0 - 1.0)'
    ],
    keyFiles: ['core/signal_calculators.py'],
    keyFunctions: ['get_random_forest_signal()']
  },
  {
    name: 'Decision Matrix Voting System',
    path: 'modules/decision_matrix',
    description: 'Weighted voting system that combines all indicator votes with accuracy-based weights',
    inputs: [
      'All indicator votes (from ATC, Range Oscillator, SPC, XGBoost, HMM, Random Forest)',
      'Indicator accuracies (historical accuracy scores)',
      'Voting threshold (default: 0.5)',
      'Min votes (minimum number of indicators required, default: 3)'
    ],
    outputs: [
      'Cumulative vote (LONG, SHORT, NONE)',
      'Weighted score (0.0 - 1.0)',
      'Voting breakdown (per-indicator contribution)',
      'Final filtered symbols (list of symbols with signals)'
    ],
    keyFiles: ['modules/decision_matrix/classifier.py', 'core/voting_analyzer.py'],
    keyFunctions: ['apply_voting_system()', 'calculate_cumulative_vote()', 'DecisionMatrixClassifier']
  }
]

// Module data for Hybrid Analyzer
const hybridModules: ModuleData[] = [
  {
    name: 'ATC Analyzer',
    path: 'modules/adaptive_trend',
    description: 'Adaptive Trend Classification - Market trend detection using KAMA-based indicators',
    inputs: [
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Timeframe (m15, m30, h1, h4, d1...)',
      'KAMA parameters (fast_period, slow_period, signal_period)'
    ],
    outputs: [
      'LONG signals (buy signals)',
      'SHORT signals (sell signals)',
      'Trend classification (UPTREND, DOWNTREND, SIDEWAYS)',
      'Signal strength (0.0 - 1.0)'
    ],
    keyFiles: ['modules/adaptive_trend/cli/main.py', 'modules/adaptive_trend/core/analyzer.py'],
    keyFunctions: ['run_auto_scan()', 'classify_trend()', 'calculate_kama()']
  },
  {
    name: 'Range Oscillator (Early Filter)',
    path: 'modules/range_oscillator',
    description: 'Early filtering step to reduce false positives before calculating other indicators',
    inputs: [
      'ATC signals (from previous step)',
      'OHLCV data (Open, High, Low, Close, Volume)',
      'Oscillator parameters (length: 14, multiplier: 2.0)'
    ],
    outputs: [
      'Filtered symbols (symbols that passed the filter)',
      'Oscillator confirmation (boolean flag)',
      'Fallback flag (true if filter too strict, fallback to ATC only)'
    ],
    keyFiles: ['modules/range_oscillator/core/oscillator.py', 'core/hybrid_analyzer.py'],
    keyFunctions: ['filter_by_oscillator()', 'filter_signals_by_range_oscillator()']
  },
  {
    name: 'SPC (Simplified Percentile Clustering)',
    path: 'modules/simplified_percentile_clustering',
    description: 'Calculate SPC signals for symbols that passed Range Oscillator filter',
    inputs: [
      'Filtered symbols (from Range Oscillator step)',
      'OHLCV data (Open, High, Low, Close, Volume)',
      'SPC parameters (k: 2, lookback: 100, p_low: 0.1, p_high: 0.9)'
    ],
    outputs: [
      'SPC signals from 3 strategies (cluster_transition, regime_following, mean_reversion)',
      'Aggregated SPC vote (LONG, SHORT, NONE)'
    ],
    keyFiles: ['modules/simplified_percentile_clustering/core/clustering.py', 'core/hybrid_analyzer.py'],
    keyFunctions: ['calculate_spc_signals_for_all()', 'calculate_spc_signals()']
  },
  {
    name: 'Decision Matrix Voting (Optional)',
    path: 'modules/decision_matrix',
    description: 'Optional voting system applied after sequential filtering',
    inputs: [
      'Filtered symbols with all signals (from previous steps)',
      'Voting threshold (default: 0.5)'
    ],
    outputs: [
      'Final filtered symbols (list of symbols with confirmed signals)',
      'Voting metadata (breakdown of votes per symbol)'
    ],
    keyFiles: ['modules/decision_matrix/classifier.py', 'core/hybrid_analyzer.py'],
    keyFunctions: ['filter_by_decision_matrix()', 'apply_decision_matrix()']
  }
]

// Handle node click in diagram
function handleNodeClick(nodeText: string, nodeMap: NodeMap, moduleDetailsRef: any) {
  debugLog('handleNodeClick called with:', nodeText)

  // Find matching module name from node text
  let moduleName: string | null = null

  // Normalize node text for comparison (remove line breaks, extra spaces)
  const normalizedNodeText = nodeText
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()

  // Try exact match first (case-insensitive)
  for (const [key, value] of Object.entries(nodeMap)) {
    if (key.toLowerCase() === normalizedNodeText) {
      moduleName = value
      break
    }
  }

  // If no exact match, try partial match
  if (!moduleName) {
    for (const [key, value] of Object.entries(nodeMap)) {
      const normalizedKey = key.toLowerCase().replace(/\s+/g, ' ')
      // Check if normalized text contains key or vice versa
      if (normalizedNodeText.includes(normalizedKey) || normalizedKey.includes(normalizedNodeText)) {
        moduleName = value
        break
      }
    }
  }

  // If still no match, try matching individual significant words
  if (!moduleName) {
    const words = normalizedNodeText.split(' ').filter(w => w.length > 2)
    for (const [key, value] of Object.entries(nodeMap)) {
      const normalizedKey = key.toLowerCase().replace(/\s+/g, ' ')
      const keyWords = normalizedKey.split(' ').filter(w => w.length > 2)
      // Check if at least 2 significant words match
      const matchingWords = words.filter(w => keyWords.includes(w))
      if (matchingWords.length >= 2 || (matchingWords.length === 1 && words.length <= 3)) {
        moduleName = value
        break
      }
    }
  }

  // Skip if moduleName is null (intentionally skipped nodes)
  if (moduleName === null) {
    debugLog('Node intentionally skipped:', nodeText)
    return
  }

  if (moduleName) {
    debugLog('Found module for node:', nodeText, '->', moduleName)
    if (moduleDetailsRef && moduleDetailsRef.value) {
      // Highlight the module
      highlightedModule.value = moduleName

      // Scroll to module
      debugLog('Calling scrollToModule on:', moduleDetailsRef.value)
      // @ts-ignore
      moduleDetailsRef.value.scrollToModule(moduleName)

      // Remove highlight after 3 seconds
      const timeoutId = setTimeout(() => {
        highlightedModule.value = null
      }, 3000)
      timeoutIds.push(timeoutId)
    } else {
      debugWarn('moduleDetailsRef.value is null or undefined', {
        moduleDetailsRef,
        hasValue: moduleDetailsRef?.value !== undefined
      })
    }
  } else {
    debugLog('No module found for node:', nodeText)
    debugLog('Normalized text:', normalizedNodeText)
    debugLog('Available keys:', Object.keys(nodeMap))
  }
}

// Add click handlers to SVG nodes
function addClickHandlersToDiagram(element: HTMLElement, nodeMap: NodeMap, moduleDetailsRef: any) {
  if (!element) {
    debugWarn('addClickHandlersToDiagram: element is null')
    return
  }

  const svgElement = element.querySelector('svg')
  if (!svgElement) {
    debugWarn('addClickHandlersToDiagram: SVG element not found')
    return
  }

  // Clean up any existing listeners before adding new ones
  cleanupFunctions.forEach(fn => fn())
  cleanupFunctions.length = 0

  // Try multiple selectors to find nodes (Mermaid may use different structures)
  let nodeGroups = svgElement.querySelectorAll('g.node')

  debugLog('Found', nodeGroups.length, 'nodes with g.node selector')

  // If no g.node found, try finding groups with shapes directly
  if (nodeGroups.length === 0) {
    debugLog('No g.node found, trying alternative selectors...')
    // Find all groups that contain shapes (don't require text, as text might be in foreignObject)
    const allGroups = svgElement.querySelectorAll('g')
    nodeGroups = Array.from(allGroups).filter(g => {
      const hasShape = g.querySelector('rect, ellipse, polygon, path, circle')
      // Don't require text here - text might be in foreignObject or elsewhere
      return hasShape
    }) as any
    debugLog('Found', nodeGroups.length, 'groups with shapes (alternative method)')
  }

  // Also try to find by class names that Mermaid might use
  if (nodeGroups.length === 0) {
    debugLog('Trying to find by class names...')
    const classBasedNodes = svgElement.querySelectorAll('g[class*="node"], g[class*="Node"]')
    if (classBasedNodes.length > 0) {
      nodeGroups = classBasedNodes as any
      debugLog('Found', nodeGroups.length, 'nodes by class name')
    }
  }

  debugLog('Total node groups found:', nodeGroups.length)

  if (nodeGroups.length === 0) {
    debugWarn('No clickable nodes found. SVG structure sample:', svgElement.innerHTML.substring(0, 500))
    return
  }

  nodeGroups.forEach((group: any, index: number) => {
    // Try multiple ways to find text in Mermaid SVG
    let nodeText = ''

    // Method 1: Look for text elements directly in group
    const textElements = group.querySelectorAll('text')

    // Method 2: Look for foreignObject (Mermaid sometimes uses this for HTML labels)
    const foreignObjects = group.querySelectorAll('foreignObject')

    // Method 3: Look for title attribute (fallback)
    const title = group.querySelector('title')

    // Method 4: Check if group has textContent directly
    const directText = group.textContent

    // Try to extract text from text elements
    if (textElements.length > 0) {
      nodeText = Array.from(textElements)
        .map((el: any) => {
          // Get all text including from tspan children
          return el.textContent || el.innerText || ''
        })
        .filter((text: any) => text.length > 0)
        .join(' ')
    }

    // If no text from text elements, try foreignObject
    if (!nodeText && foreignObjects.length > 0) {
      nodeText = Array.from(foreignObjects)
        .map((fo: any) => {
          // Get text from foreignObject (may contain HTML)
          const textEl = fo.querySelector('p, div, span')
          return textEl ? (textEl.textContent || textEl.innerText || '') : ''
        })
        .filter((text: any) => text.length > 0)
        .join(' ')
    }

    // If still no text, try title
    if (!nodeText && title) {
      nodeText = title.textContent || (title as HTMLElement).innerText || ''
    }

    // Last resort: use direct textContent
    if (!nodeText && directText) {
      nodeText = directText
    }

    // Clean up the text
    nodeText = nodeText
      .replace(/\n/g, ' ')
      .replace(/\r/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()

    // Skip if text is empty or too short
    if (!nodeText || nodeText.length < 2) {
      debugLog(`Node ${index}: No text found. Methods tried:`, {
        textElements: textElements.length,
        foreignObjects: foreignObjects.length,
        title: !!title,
        directText: directText?.substring(0, 50)
      })
      // Log the group structure for debugging
      debugLog(`Node ${index} structure:`, group.innerHTML.substring(0, 200))
      return
    }

    debugLog(`Node ${index}: "${nodeText}"`)

    // Find the shape element (rect, ellipse, polygon, path, circle)
    const shape = group.querySelector('rect, ellipse, polygon, path, circle') as HTMLElement
    if (!shape) {
      debugLog(`Node ${index}: No shape element found`)
      return
    }

    // Add cursor pointer and hover effects
    shape.style.cursor = 'pointer'
    shape.style.transition = 'all 0.2s ease'

    // Store original styles for hover effect
    const originalOpacity = shape.style.opacity || '1'
    const originalFilter = shape.style.filter || ''

    // Create named handlers for cleanup
    const handleMouseEnter = () => {
      shape.style.opacity = '0.9'
      shape.style.filter = 'brightness(1.2) drop-shadow(0 0 8px rgba(139, 92, 246, 0.6))'
    }

    const handleMouseLeave = () => {
      shape.style.opacity = originalOpacity
      shape.style.filter = originalFilter
    }

    const handleClick = (e: Event) => {
      e.stopPropagation()
      e.preventDefault()
      debugLog('Node clicked:', nodeText)
      handleNodeClick(nodeText, nodeMap, moduleDetailsRef)
    }

    // Add hover effect with glow
    group.addEventListener('mouseenter', handleMouseEnter)
    group.addEventListener('mouseleave', handleMouseLeave)
    group.addEventListener('click', handleClick)

    // Store cleanup functions
    cleanupFunctions.push(() => {
      group.removeEventListener('mouseenter', handleMouseEnter)
      group.removeEventListener('mouseleave', handleMouseLeave)
      group.removeEventListener('click', handleClick)
    })

      // Make the entire group clickable
      (group as HTMLElement).style.cursor = 'pointer'

    // Also make text elements clickable (if they exist)
    if (textElements.length > 0) {
      textElements.forEach((textEl: any) => {
        textEl.style.cursor = 'pointer'
        textEl.style.pointerEvents = 'auto'
      })
    }

    // Make foreignObjects clickable too
    if (foreignObjects.length > 0) {
      foreignObjects.forEach((fo: any) => {
        fo.style.cursor = 'pointer'
        fo.style.pointerEvents = 'auto'
      })
    }
  })

  debugLog('Click handlers added to', nodeGroups.length, 'nodes')
}

// Render diagrams
async function renderDiagram(element: HTMLElement, diagramCode: string, nodeMap: NodeMap, moduleDetailsRef: any) {
  if (!element) {
    debugWarn('Render diagram: element is null')
    return
  }

  // Ensure mermaid is initialized
  initializeMermaid()

  try {
    // Clear previous content
    element.innerHTML = '<div class="text-gray-400 p-4 text-center">Loading diagram...</div>'

    // Generate unique ID
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`

    debugLog('Rendering Mermaid diagram with ID:', id)

    // Use mermaid.render() for version 10.x
    const result = await mermaid.render(id, diagramCode)

    if (!result || !result.svg) {
      throw new Error('Mermaid render returned no SVG')
    }

    // Set the SVG content
    element.innerHTML = result.svg

    // Apply styling to the SVG
    const svgElement = element.querySelector('svg')
    if (svgElement) {
      svgElement.style.display = 'block'
      svgElement.style.margin = '0 auto'

      // Wait for SVG to fully render - don't modify text positions, let Mermaid handle it
      await nextTick()
      await new Promise(resolve => setTimeout(resolve, 100))

      // Only ensure text is visible - don't modify positions
      const ensureTextVisible = () => {
        const nodes = svgElement.querySelectorAll('.node')
        debugLog('Ensuring text visible for', nodes.length, 'nodes')

        nodes.forEach((node, index) => {
          // Ensure all text elements are visible
          const textElements = node.querySelectorAll('text')
          debugLog(`Node ${index}: Found ${textElements.length} text elements`)

          textElements.forEach((textEl: any) => {
            const textContent = textEl.textContent?.trim() || ''
            if (textContent) {
              debugLog(`  Text element: "${textContent}"`)
            }
            textEl.style.fill = 'currentColor'
            textEl.style.opacity = '1'
            textEl.style.visibility = 'visible'
            textEl.style.display = 'block'
            // Don't modify x or text-anchor - let Mermaid handle it
          })

          // Ensure tspan elements are visible
          const tspans = node.querySelectorAll('tspan')
          tspans.forEach((tspan: any) => {
            tspan.style.fill = 'currentColor'
            tspan.style.opacity = '1'
            tspan.style.visibility = 'visible'
            tspan.style.display = 'block'
          })

          // Ensure foreignObject content is visible (for htmlLabels: true)
          const foreignObjects = node.querySelectorAll('foreignObject')
          debugLog(`Node ${index}: Found ${foreignObjects.length} foreignObject elements`)

          foreignObjects.forEach((fo: any) => {
            fo.style.visibility = 'visible'
            fo.style.opacity = '1'
            fo.style.display = 'block'
            const div = fo.querySelector('div')
            if (div) {
              const divText = div.textContent?.trim() || ''
              if (divText) {
                debugLog(`  ForeignObject div: "${divText}"`)
              }
              div.style.visibility = 'visible'
              div.style.opacity = '1'
              div.style.display = 'block'
              div.style.color = 'inherit'
            }
          })
        })
      }

      // Ensure text is visible - run once after render
      await nextTick()
      await new Promise(resolve => setTimeout(resolve, 200)) // Wait longer for Mermaid to fully render
      ensureTextVisible()

      // Recalculate after node adjustments
      await nextTick()
      const container = element.closest('.mermaid-container') as HTMLElement
      const containerWidth = container ? container.offsetWidth - 40 : 800 // Subtract padding

      // @ts-ignore
      const svgWidth = svgElement.getBBox().width || svgElement.viewBox.baseVal.width
      // @ts-ignore
      const svgHeight = svgElement.getBBox().height || svgElement.viewBox.baseVal.height

      if (svgWidth && svgHeight && containerWidth) {
        // Set scale to 1 (no scaling)
        svgElement.style.transform = 'scale(1)'
        svgElement.style.transformOrigin = 'center top'

        // Adjust container height to accommodate diagram
        element.style.minHeight = '650px'
        element.style.width = '100%'
        element.style.maxWidth = '100%'
        svgElement.style.maxWidth = '100%'
      } else {
        // Fallback: use scale 1
        svgElement.style.transform = 'scale(1)'
        svgElement.style.transformOrigin = 'center top'
        svgElement.style.maxWidth = '100%'
      }
    }

    // Add click handlers after a short delay to ensure DOM is ready
    await nextTick()
    const timeoutId = setTimeout(() => {
      debugLog('Adding click handlers, moduleDetailsRef:', moduleDetailsRef?.value)
      addClickHandlersToDiagram(element, nodeMap, moduleDetailsRef)
    }, 300)
    timeoutIds.push(timeoutId)

    debugLog('Diagram rendered successfully')
  } catch (error: any) {
    console.error('Error rendering Mermaid diagram:', error)
    console.error('Error details:', {
      message: error.message,
      stack: error.stack,
      diagramCode: diagramCode.substring(0, 100) + '...'
    })

    if (element) {
      const errorDiv = document.createElement('div')
      errorDiv.className = 'text-red-400 p-4 border border-red-500/50 rounded'

      const title = document.createElement('p')
      title.className = 'font-semibold'
      title.textContent = 'Error rendering diagram:'

      const message = document.createElement('p')
      message.className = 'text-sm text-gray-400 mt-2'
      message.textContent = error.message

      const hint = document.createElement('p')
      hint.className = 'text-xs text-gray-500 mt-2'
      hint.textContent = 'Check browser console for details'

      errorDiv.appendChild(title)
      errorDiv.appendChild(message)
      errorDiv.appendChild(hint)

      element.innerHTML = ''
      element.appendChild(errorDiv)
    }
  }
}

// Watch for tab changes and render appropriate diagram
watch(activeTab, async (newTab) => {
  await nextTick()
  highlightedModule.value = null // Clear highlight when switching tabs
  if (newTab === 'voting' && votingDiagram.value) {
    renderDiagram(votingDiagram.value, votingDiagramCode, votingNodeToModuleMap, votingModuleDetails)
  } else if (newTab === 'hybrid' && hybridDiagram.value) {
    renderDiagram(hybridDiagram.value, hybridDiagramCode, hybridNodeToModuleMap, hybridModuleDetails)
  }
})

// Initial render
onMounted(async () => {
  await nextTick()
  if (activeTab.value === 'voting' && votingDiagram.value) {
    renderDiagram(votingDiagram.value, votingDiagramCode, votingNodeToModuleMap, votingModuleDetails)
  } else if (activeTab.value === 'hybrid' && hybridDiagram.value) {
    renderDiagram(hybridDiagram.value, hybridDiagramCode, hybridNodeToModuleMap, hybridModuleDetails)
  }
})

// Cleanup event listeners and timeouts on unmount
onUnmounted(() => {
  // Cleanup event listeners
  cleanupFunctions.forEach(fn => fn())
  cleanupFunctions.length = 0

  // Cleanup timeouts
  timeoutIds.forEach(id => clearTimeout(id))
  timeoutIds.length = 0
})
</script>

<style scoped>
.mermaid-container {
  min-height: 350px;
  overflow-y: visible;
  overflow-x: hidden;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
}

.mermaid-diagram {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 15px;
  min-height: 350px;
  width: 100%;
  max-width: 100%;
  overflow: hidden;
}

.mermaid-diagram :deep(svg) {
  height: auto;
  width: auto;
  max-width: 100%;
  overflow: visible;
}

/* Optimized font sizes - increased by 30% and bold */
.mermaid-diagram :deep(.node) {
  font-size: 17px;
  font-weight: bold;
  min-width: 120px;
  padding: 12px 16px;
}

.mermaid-diagram :deep(.nodeLabel) {
  font-size: 17px;
  font-weight: bold;
  line-height: 1.4;
  padding: 8px 12px;
  white-space: normal;
  word-wrap: break-word;
}

.mermaid-diagram :deep(.edgeLabel) {
  font-size: 11px;
}

/* Also target text elements directly */
.mermaid-diagram :deep(text) {
  font-weight: bold;
}

.mermaid-diagram :deep(.node text) {
  font-size: 17px;
  font-weight: bold;
  fill: currentColor !important;
  opacity: 1 !important;
  visibility: visible !important;
  display: block !important;
}

.mermaid-diagram :deep(.node tspan) {
  fill: currentColor !important;
  opacity: 1 !important;
  visibility: visible !important;
  display: block !important;
}

.mermaid-diagram :deep(.node foreignObject) {
  opacity: 1 !important;
  visibility: visible !important;
}

.mermaid-diagram :deep(.node foreignObject div) {
  opacity: 1 !important;
  visibility: visible !important;
  display: block !important;
}

/* Ensure nodes have enough space for text */
.mermaid-diagram :deep(.node rect),
.mermaid-diagram :deep(.node polygon) {
  min-width: 120px;
  overflow: visible;
}

/* Responsive scaling */
@media (max-width: 1024px) {
  .mermaid-diagram :deep(svg) {
    transform: scale(0.9) !important;
    transform-origin: center top;
  }
}

@media (max-width: 768px) {
  .mermaid-diagram :deep(svg) {
    transform: scale(0.8) !important;
    transform-origin: center top;
  }
}
</style>

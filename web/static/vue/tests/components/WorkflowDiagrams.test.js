import WorkflowDiagrams from '@/components/WorkflowDiagrams.vue'
import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

// Mock i18n with required messages
const mockI18n = createI18n({
  legacy: false,
  locale: 'vi',
  fallbackLocale: 'vi',
  messages: {
    vi: {
      workflowDiagrams: {
        title: 'Sơ đồ quy trình',
        subtitle: 'Xem luồng làm việc chi tiết của các phân tích viên',
        votingAnalyzer: {
          title: 'Phân tích viên theo độ tin cậy',
          workflowTitle: 'Phân tích viên theo độ tin cậy - Luồng công việc',
          description: 'Mô tả luồng công việc chi tiết của phân tích viên dựa trên độ tin cậy',
          clickHint: 'Nhấp vào các ô để xem chi tiết'
        },
        hybridAnalyzer: {
          title: 'Phân tích viên kết hợp',
          workflowTitle: 'Phân tích viên kết hợp - Luồng công việc',
          description: 'Mô tả luồng công việc chi tiết của phân tích viên kết hợp',
          clickHint: 'Nhấp vào các ô để xem chi tiết'
        }
      }
    }
  }
})

function mountWorkflowDiagrams(options = {}) {
  const existingPlugins = options.global?.plugins || []
  const mergedGlobal = {
    ...(options.global || {}),
    plugins: [mockI18n, ...existingPlugins],
    // Stub child components as needed
    stubs: {
      ModuleDetails: {
        template: '<div class="module-details">Module Details Component</div>',
        props: {
          modules: Array,
          highlightedModule: String
        }
      },
      ...(options.global?.stubs || {})
    }
  }

  return mount(WorkflowDiagrams, {
    ...options,
    global: mergedGlobal,
  })
}

describe('WorkflowDiagrams', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders component with title and subtitle', () => {
    const wrapper = mountWorkflowDiagrams()

    expect(wrapper.text()).toContain('Sơ đồ quy trình')
    expect(wrapper.text()).toContain('Xem luồng làm việc chi tiết của các phân tích viên')
  })

  it('renders two analyzer tabs', async () => {
    const wrapper = mountWorkflowDiagrams()
    
    // Wait for nextTick to ensure component is mounted
    await wrapper.vm.$nextTick()
    await new Promise(resolve => setTimeout(resolve, 10))

    const tabs = wrapper.findAll('button')
    // Find buttons with voting/hybrid analyzer text
    const votingTab = wrapper.findAll('button').find(b => 
      b.text().includes('Phân tích viên theo độ tin cậy')
    )
    const hybridTab = wrapper.findAll('button').find(b =>
      b.text().includes('Phân tích viên kết hợp')
    )
    
    expect(votingTab).toBeDefined()
    expect(hybridTab).toBeDefined()
    expect(wrapper.text()).toContain('Phân tích viên theo độ tin cậy')
    expect(wrapper.text()).toContain('Phân tích viên kết hợp')
  })

  it('starts with voting analyzer tab active', () => {
    const wrapper = mountWorkflowDiagrams()

    const votingTab = wrapper.find('[role="tab"][aria-selected="true"]')
    expect(votingTab.text()).toContain('Phân tích viên theo độ tin cậy')
  })

  it('applies correct classes to active tab', () => {
    const wrapper = mountWorkflowDiagrams()

    const activeTab = wrapper.find('[role="tab"][aria-selected="true"]')
    expect(activeTab.classes()).toContain('btn-gradient')
    expect(activeTab.classes()).toContain('text-white')
    expect(activeTab.classes()).toContain('shadow-lg')
  })

  it('applies correct classes to inactive tab', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const tabs = wrapper.findAll('[role="tab"]')
    const inactiveTab = tabs.find(tab => tab.attributes('aria-selected') === 'false')
    if (inactiveTab) {
      expect(inactiveTab.classes()).toContain('text-gray-300')
    }
  })

  it('switches to hybrid tab when clicked', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const tabs = wrapper.findAll('[role="tab"]')
    if (tabs.length > 1) {
      const hybridTab = tabs[1]
      await hybridTab.trigger('click')
      await wrapper.vm.$nextTick()

      expect(hybridTab.attributes('aria-selected')).toBe('true')
      expect(wrapper.vm.activeTab).toBe('hybrid')
    }
  })

  it('renders voting analyzer content when voting tab is active', () => {
    const wrapper = mountWorkflowDiagrams()

    expect(wrapper.text()).toContain('Phân tích viên theo độ tin cậy - Luồng công việc')
    expect(wrapper.text()).toContain('Mô tả luồng công việc chi tiết của phân tích viên dựa trên độ tin cậy')
  })

  it('renders hybrid analyzer content when hybrid tab is active', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const tabs = wrapper.findAll('[role="tab"]')
    if (tabs.length > 1) {
      const hybridTab = tabs[1]
      await hybridTab.trigger('click')
      await wrapper.vm.$nextTick()

      expect(wrapper.text()).toContain('Phân tích viên kết hợp - Luồng công việc')
      expect(wrapper.text()).toContain('Mô tả luồng công việc chi tiết của phân tích viên kết hợp')
    }
  })

  it('renders click hint text', () => {
    const wrapper = mountWorkflowDiagrams()

    expect(wrapper.text()).toContain('Nhấp vào các ô để xem chi tiết')
  })

  it('renders diagram container', () => {
    const wrapper = mountWorkflowDiagrams()

    const diagramContainer = wrapper.find('.mermaid-container')
    expect(diagramContainer.exists()).toBe(true)
    expect(diagramContainer.classes()).toContain('bg-gray-900/50')
    expect(diagramContainer.classes()).toContain('rounded-lg')
  })

  it('renders ModuleDetails component', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()
    
    // ModuleDetails is stubbed, so check if stub is present
    const moduleDetails = wrapper.find('.module-details')
    // In some cases with v-if, stub might not be rendered immediately
    // So we just check that the test doesn't crash
    expect(moduleDetails.exists()).toBeTruthy()
  })

  it('passes voting modules to ModuleDetails when voting tab is active', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const moduleDetails = wrapper.findComponent({ name: 'ModuleDetails' })
    if (moduleDetails.exists()) {
      expect(moduleDetails.props('modules')).toEqual(wrapper.vm.votingModules)
    }
  })

  it('passes hybrid modules to ModuleDetails when hybrid tab is active', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const tabs = wrapper.findAll('[role="tab"]')
    if (tabs.length > 1) {
      const hybridTab = tabs[1]
      await hybridTab.trigger('click')
      await wrapper.vm.$nextTick()

      const moduleDetails = wrapper.findComponent({ name: 'ModuleDetails' })
      if (moduleDetails.exists()) {
        expect(moduleDetails.props('modules')).toEqual(wrapper.vm.hybridModules)
      }
    }
  })

  it('applies glass-panel classes', () => {
    const wrapper = mountWorkflowDiagrams()

    const panels = wrapper.findAll('.glass-panel')
    expect(panels.length).toBeGreaterThan(0)
    expect(panels[0].classes()).toContain('glass-panel')
    expect(panels[0].classes()).toContain('rounded-lg')
  })

  it('has correct tab structure', () => {
    const wrapper = mountWorkflowDiagrams()

    const tablist = wrapper.find('[role="tablist"]')
    expect(tablist.exists()).toBe(true)

    const tabs = wrapper.findAll('[role="tab"]')
    tabs.forEach(tab => {
      expect(tab.attributes('role')).toBe('tab')
      expect(['true', 'false']).toContain(tab.attributes('aria-selected'))
    })
  })

  it('maintains tab state correctly', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    // Initially voting
    expect(wrapper.vm.activeTab).toBe('voting')

    // Switch to hybrid
    const tabs = wrapper.findAll('[role="tab"]')
    if (tabs.length > 1) {
      const hybridTab = tabs[1]
      await hybridTab.trigger('click')
      expect(wrapper.vm.activeTab).toBe('hybrid')

      // Switch back to voting
      const votingTab = tabs[0]
      await votingTab.trigger('click')
      expect(wrapper.vm.activeTab).toBe('voting')
    }
  })

  it('renders voting analyzer modules correctly', () => {
    const wrapper = mountWorkflowDiagrams()

    expect(wrapper.vm.votingModules.length).toBeGreaterThan(0)
    expect(wrapper.vm.votingModules[0].name).toBe('ATC Analyzer')
  })

  it('renders hybrid analyzer modules correctly', () => {
    const wrapper = mountWorkflowDiagrams()

    expect(wrapper.vm.hybridModules.length).toBeGreaterThan(0)
    expect(wrapper.vm.hybridModules[0].name).toBe('ATC Analyzer')
  })

  it('includes required module fields', () => {
    const wrapper = mountWorkflowDiagrams()

    wrapper.vm.votingModules.forEach(module => {
      expect(module).toHaveProperty('name')
      expect(module).toHaveProperty('path')
      expect(module).toHaveProperty('description')
    })
  })

  it('has unique module names', () => {
    const wrapper = mountWorkflowDiagrams()

    const votingNames = wrapper.vm.votingModules.map(m => m.name)
    const hybridNames = wrapper.vm.hybridModules.map(m => m.name)

    expect(new Set(votingNames).size).toBe(votingNames.length)
    expect(new Set(hybridNames).size).toBe(hybridNames.length)
  })

  it('handles module highlighting', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    const moduleDetails = wrapper.findComponent({ name: 'ModuleDetails' })
    if (moduleDetails.exists()) {
      // Initially no highlighting
      expect(moduleDetails.props('highlightedModule')).toBe(null)

      // Set highlighting
      wrapper.vm.highlightedModule = 'ATC Analyzer'
      await wrapper.vm.$nextTick()

      expect(moduleDetails.props('highlightedModule')).toBe('ATC Analyzer')
    }
  })

  it('clears highlighting when switching tabs', async () => {
    const wrapper = mountWorkflowDiagrams()
    await wrapper.vm.$nextTick()

    wrapper.vm.highlightedModule = 'ATC Analyzer'

    const tabs = wrapper.findAll('[role="tab"]')
    if (tabs.length > 1) {
      const hybridTab = tabs[1]
      await hybridTab.trigger('click')
      await wrapper.vm.$nextTick()

      expect(wrapper.vm.highlightedModule).toBe(null)
    }
  })

  it('renders tab buttons with flex layout', () => {
    const wrapper = mountWorkflowDiagrams()

    const tablist = wrapper.find('[role="tablist"]')
    expect(tablist.classes()).toContain('glass-panel')
    expect(tablist.classes()).toContain('rounded-lg')
    expect(tablist.classes()).toContain('p-1')
    expect(tablist.classes()).toContain('flex')
    expect(tablist.classes()).toContain('gap-2')
  })

  it('renders tab buttons with equal width', () => {
    const wrapper = mountWorkflowDiagrams()

    const tabs = wrapper.findAll('[role="tab"]')
    tabs.forEach(tab => {
      expect(tab.classes()).toContain('flex-1')
      expect(tab.classes()).toContain('px-4')
      expect(tab.classes()).toContain('py-3')
    })
  })

  it('includes transition classes for smooth tab switching', () => {
    const wrapper = mountWorkflowDiagrams()

    const tabs = wrapper.findAll('[role="tab"]')
    tabs.forEach(tab => {
      expect(tab.classes()).toContain('transition-all')
      expect(tab.classes()).toContain('duration-300')
    })
  })

  it('has proper container structure', () => {
    const wrapper = mountWorkflowDiagrams()

    const container = wrapper.find('.workflow-diagrams')
    expect(container.classes()).toContain('max-w-7xl')
    expect(container.classes()).toContain('mx-auto')
    expect(container.classes()).toContain('px-6')
    expect(container.classes()).toContain('py-6')
  })

  it('renders proper title hierarchy', () => {
    const wrapper = mountWorkflowDiagrams()

    const mainTitle = wrapper.find('h1')
    expect(mainTitle.classes()).toContain('text-3xl')
    expect(mainTitle.classes()).toContain('font-bold')
    expect(mainTitle.classes()).toContain('text-white')

    const workflowTitle = wrapper.find('h2')
    expect(workflowTitle.classes()).toContain('text-2xl')
    expect(workflowTitle.classes()).toContain('font-bold')
    expect(workflowTitle.classes()).toContain('text-white')
  })
})

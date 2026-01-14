import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ModuleDetails from '../../src/components/ModuleDetails.vue'

const mockModules = [
  {
    name: 'ATC Analyzer',
    path: 'modules/adaptive_trend',
    description: 'Adaptive Trend Classification',
    inputs: ['OHLCV data', 'Timeframe'],
    outputs: ['LONG signals', 'SHORT signals'],
    keyFiles: ['main.py'],
    keyFunctions: ['run_auto_scan()']
  },
  {
    name: 'Range Oscillator',
    path: 'modules/range_oscillator',
    description: 'Overbought/oversold detection',
    inputs: ['OHLCV data'],
    outputs: ['Oscillator signal']
  }
]

function mountModuleDetails(options = {}) {
  return mount(ModuleDetails, {
    ...options,
    global: {
      mocks: {
        $t: (key) => key === 'workflowDiagrams.moduleDetails.title' ? 'Chi tiết mô-đun' : key
      }
    }
  })
}

describe('ModuleDetails', () => {
  it('renders with modules', () => {
    const wrapper = mountModuleDetails({
      props: { modules: mockModules }
    })

    expect(wrapper.text()).toContain('ATC Analyzer')
    expect(wrapper.text()).toContain('Range Oscillator')
  })

  it('starts with modules collapsed', () => {
    const wrapper = mountModuleDetails({
      props: { modules: mockModules }
    })

    expect(wrapper.find('.mt-4').exists()).toBe(false)
  })

  it('expands module when clicked', async () => {
    const wrapper = mountModuleDetails({
      props: { modules: mockModules }
    })

    const moduleHeaders = wrapper.findAll('.cursor-pointer')
    await moduleHeaders[0].trigger('click')

    expect(wrapper.find('.mt-4').exists()).toBe(true)
  })

  it('shows module details when expanded', async () => {
    const wrapper = mountModuleDetails({
      props: { modules: mockModules }
    })

    const moduleHeaders = wrapper.findAll('.cursor-pointer')
    await moduleHeaders[0].trigger('click')

    expect(wrapper.text()).toContain('modules/adaptive_trend')
    expect(wrapper.text()).toContain('Adaptive Trend Classification')
  })

  it('applies highlighted border when highlighted', () => {
    const wrapper = mountModuleDetails({
      props: {
        modules: mockModules,
        highlightedModule: 'ATC Analyzer'
      }
    })

    const firstModule = wrapper.findAll('[id^="module-"]')[0]
    expect(firstModule.classes()).toContain('border-purple-500')
  })

  it('renders glass-panel classes', () => {
    const wrapper = mountModuleDetails({
      props: { modules: mockModules }
    })

    const panels = wrapper.findAll('.glass-panel')
    expect(panels.length).toBe(2)
  })

  it('handles empty modules array', () => {
    const wrapper = mountModuleDetails({
      props: { modules: [] }
    })

    expect(wrapper.find('[id^="module-"]').exists()).toBe(false)
  })
})
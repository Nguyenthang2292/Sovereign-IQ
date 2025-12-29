/**
 * Tests for LogViewer component
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { i18n } from '../setup'
import LogViewer from '../../src/components/LogViewer.vue'

// Helper to mount with i18n
function mountLogViewer(options = {}) {
  const existingPlugins = options.global?.plugins || []
  const mergedGlobal = {
    ...(options.global || {}),
    plugins: [i18n, ...existingPlugins],
  }
  
  return mount(LogViewer, {
    ...options,
    global: mergedGlobal,
  })
}

describe('LogViewer', () => {
  it('should render empty state when no logs', () => {
    const wrapper = mountLogViewer({
      props: {
        logs: [],
      },
    })

    expect(wrapper.text()).toContain('Đang chờ logs...')
    expect(wrapper.text()).toContain('⏳')
  })

  it('should render logs', () => {
    const logs = ['Log 1', 'Log 2', 'Log 3']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    expect(wrapper.text()).toContain('Log 1')
    expect(wrapper.text()).toContain('Log 2')
    expect(wrapper.text()).toContain('Log 3')
  })

  it('should detect and display error logs with error styling', () => {
    const logs = ['Error occurred', 'Failed to process']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    const logElements = wrapper.findAll('.text-red-400')
    expect(logElements.length).toBeGreaterThan(0)
    
    // Check for error icon
    expect(wrapper.text()).toContain('❌')
  })

  it('should detect and display warning logs with warning styling', () => {
    const logs = ['Warning message', 'Caution needed']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    const logElements = wrapper.findAll('.text-yellow-400')
    expect(logElements.length).toBeGreaterThan(0)
    
    // Check for warning icon
    expect(wrapper.text()).toContain('⚠️')
  })

  it('should detect and display success logs with success styling', () => {
    const logs = ['Success!', 'Created file', 'Completed task']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    const logElements = wrapper.findAll('.text-green-400')
    expect(logElements.length).toBeGreaterThan(0)
    
    // Check for success icon
    expect(wrapper.text()).toContain('✅')
  })

  it('should detect and display info logs with info styling', () => {
    const logs = ['Analyzing data', 'Sending request', 'Processing items']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    const logElements = wrapper.findAll('.text-blue-400')
    expect(logElements.length).toBeGreaterThan(0)
    
    // Check for info icon
    expect(wrapper.text()).toContain('ℹ️')
  })

  it('should parse and display ANSI color codes', () => {
    const logs = ['\x1b[34mBlue text\x1b[0m', '\x1b[32mGreen text\x1b[0m']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    // Check that ANSI codes are parsed and displayed
    expect(wrapper.text()).toContain('Blue text')
    expect(wrapper.text()).toContain('Green text')
    
    // Check for color classes
    const blueElements = wrapper.findAll('.text-blue-400')
    const greenElements = wrapper.findAll('.text-green-400')
    expect(blueElements.length).toBeGreaterThan(0)
    expect(greenElements.length).toBeGreaterThan(0)
  })
  
  it('should handle full ANSI escape sequences', () => {
    const logs = ['\x1b[34mHello\x1b[0m World']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    expect(wrapper.text()).toContain('Hello')
    expect(wrapper.text()).toContain('World')
  })

  it('should apply correct container classes for different log levels', () => {
    const logs = [
      'Error occurred',
      'Warning message',
      'Success!',
      'Analyzing data',
      '[2024-01-01] Debug log',
    ]
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    // Check for error container class - classes are applied to <li> elements
    // The classes are: bg-red-900/10 border-l-2 border-red-500
    const allLi = wrapper.findAll('li')
    expect(allLi.length).toBeGreaterThan(0)
    
    // Check if at least one li has error-related classes
    const errorItems = allLi.filter(li => {
      const classes = li.classes()
      return classes.some(c => c.includes('bg-red-900') || c.includes('border-red-500'))
    })
    expect(errorItems.length).toBeGreaterThan(0)

    // Check for warning container class
    const warningItems = allLi.filter(li => {
      const classes = li.classes()
      return classes.some(c => c.includes('bg-yellow-900') || c.includes('border-yellow-500'))
    })
    expect(warningItems.length).toBeGreaterThan(0)

    // Check for success container class
    const successItems = allLi.filter(li => {
      const classes = li.classes()
      return classes.some(c => c.includes('bg-green-900') || c.includes('border-green-500'))
    })
    expect(successItems.length).toBeGreaterThan(0)

    // Check for info container class
    const infoItems = allLi.filter(li => {
      const classes = li.classes()
      return classes.some(c => c.includes('bg-blue-900') || c.includes('border-blue-500'))
    })
    expect(infoItems.length).toBeGreaterThan(0)
  })

  it('should have correct CSS classes for styling', () => {
    const wrapper = mountLogViewer({
      props: {
        logs: ['Test log'],
      },
    })

    // Check container classes
    const container = wrapper.find('.bg-gray-900')
    expect(container.exists()).toBe(true)
    expect(container.classes()).toContain('rounded-lg')
    expect(container.classes()).toContain('border')
    expect(container.classes()).toContain('border-gray-700')
    expect(container.classes()).toContain('max-h-96')
    expect(container.classes()).toContain('overflow-y-auto')
  })

  it('should render logs with monospace font', () => {
    const wrapper = mountLogViewer({
      props: {
        logs: ['Test log'],
      },
    })

    const logContainer = wrapper.find('.font-mono')
    expect(logContainer.exists()).toBe(true)
  })

  it('should handle mixed log types', () => {
    const logs = [
      'Error occurred',
      'Warning message',
      'Success!',
      'Analyzing data',
      'Plain log message',
    ]
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    // All logs should be rendered
    logs.forEach((log) => {
      expect(wrapper.text()).toContain(log)
    })
  })

  it('should handle logs with special characters', () => {
    const logs = [
      'Log with émojis 🎉',
      'Log with "quotes"',
      "Log with 'apostrophes'",
      'Log with <tags>',
      'Log with & symbols',
    ]
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    logs.forEach((log) => {
      expect(wrapper.text()).toContain(log)
    })
  })

  it('should handle very long log messages', () => {
    const longLog = 'A'.repeat(500)
    const wrapper = mountLogViewer({
      props: {
        logs: [longLog],
      },
    })

    expect(wrapper.text()).toContain('A')
  })

  it('should handle logs with newlines', () => {
    const logs = ['Line 1\nLine 2\nLine 3']
    const wrapper = mountLogViewer({
      props: {
        logs,
      },
    })

    // Newlines should be preserved in text content
    expect(wrapper.text()).toContain('Line 1')
    expect(wrapper.text()).toContain('Line 2')
    expect(wrapper.text()).toContain('Line 3')
  })

  it('should update when logs prop changes', async () => {
    const wrapper = mountLogViewer({
      props: {
        logs: ['Initial log'],
      },
    })

    expect(wrapper.text()).toContain('Initial log')

    await wrapper.setProps({
      logs: ['Updated log 1', 'Updated log 2'],
    })

    expect(wrapper.text()).toContain('Updated log 1')
    expect(wrapper.text()).toContain('Updated log 2')
    expect(wrapper.text()).not.toContain('Initial log')
  })
})


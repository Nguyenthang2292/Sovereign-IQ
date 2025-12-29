/**
 * Tests for LoadingSpinner component
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import LoadingSpinner from '../../src/components/LoadingSpinner.vue'

describe('LoadingSpinner', () => {
  it('should render spinner without message', () => {
    const wrapper = mount(LoadingSpinner)

    expect(wrapper.find('svg').exists()).toBe(true)
    expect(wrapper.find('span').exists()).toBe(false)
  })

  it('should render spinner with message', () => {
    const message = 'Loading data...'
    const wrapper = mount(LoadingSpinner, {
      props: {
        message,
      },
    })

    expect(wrapper.find('svg').exists()).toBe(true)
    expect(wrapper.find('span').text()).toBe(message)
  })

  it('should have proper accessibility attributes', () => {
    const wrapper = mount(LoadingSpinner)

    const container = wrapper.find('div')
    // Verify container has appropriate ARIA role for screen readers
    expect(container.attributes('role')).toBe('status')
    expect(container.attributes('aria-label')).toBe('Loading')

    // Verify SVG spinner element exists
    const svg = wrapper.find('svg')
    expect(svg.exists()).toBe(true)
    
    // Verify SVG is hidden from screen readers since we have aria-label on container
    expect(svg.attributes('aria-hidden')).toBe('true')
  })

  it('should have proper accessibility attributes when message prop is provided', () => {
    const message = 'Loading data...'
    const wrapper = mount(LoadingSpinner, {
      props: {
        message,
      },
    })

    const container = wrapper.find('div')
    // Verify container has appropriate ARIA role for screen readers
    expect(container.attributes('role')).toBe('status')
    expect(container.attributes('aria-live')).toBe('polite')
    // Verify aria-label reflects the provided message prop
    expect(container.attributes('aria-label')).toBe(message)

    // Verify SVG spinner element exists and has proper structure
    const svg = wrapper.find('svg')
    expect(svg.exists()).toBe(true)
    
    // Verify SVG contains spinner elements (circle and path) to ensure accessibility structure remains intact
    expect(svg.find('circle').exists()).toBe(true)
    expect(svg.find('path').exists()).toBe(true)
  })

  it('should have aria-live on message span when message is provided', () => {
    const message = 'Loading data...'
    const wrapper = mount(LoadingSpinner, {
      props: {
        message,
      },
    })

    const span = wrapper.find('span')
    expect(span.exists()).toBe(true)
    expect(span.attributes('aria-live')).toBe('polite')
  })
})


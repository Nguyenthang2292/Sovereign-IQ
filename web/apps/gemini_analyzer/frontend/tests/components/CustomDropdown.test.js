import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { i18n } from '../setup'
import CustomDropdown from '../../src/components/CustomDropdown.vue'

vi.mock('../../src/i18n/locales/vi.json', () => ({
  default: {}
}))

vi.mock('../../src/i18n/locales/en.json', () => ({
  default: {}
}))

function mountCustomDropdown(options = {}) {
  const existingPlugins = options.global?.plugins || []
  const mergedGlobal = {
    ...(options.global || {}),
    plugins: [i18n, ...existingPlugins],
  }

  return mount(CustomDropdown, {
    ...options,
    global: mergedGlobal,
  })
}

describe('CustomDropdown', () => {
  it('renders with basic props', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['Option 1', 'Option 2', 'Option 3'],
        placeholder: 'Select option'
      }
    })

    expect(wrapper.find('.custom-dropdown').exists()).toBe(true)
    expect(wrapper.text()).toContain('Select option')
  })

  it('displays selected value', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['Option 1', 'Option 2', 'Option 3'],
        modelValue: 'Option 2'
      }
    })

    expect(wrapper.text()).toContain('Option 2')
  })

  it('handles object options correctly', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { label: 'First Option', value: 'first' },
          { label: 'Second Option', value: 'second' }
        ],
        modelValue: 'second'
      }
    })

    expect(wrapper.text()).toContain('Second Option')
  })

  it('emits update:modelValue when option is selected', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C']
      }
    })

    // Trigger the selectOption method directly
    wrapper.vm.selectOption('B', 1)

    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
    expect(wrapper.emitted('update:modelValue')[0]).toEqual(['B'])
  })

  it('emits change event when option is selected', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C']
      }
    })

    wrapper.vm.selectOption('C', 2)

    expect(wrapper.emitted('change')).toBeTruthy()
    expect(wrapper.emitted('change')[0]).toEqual(['C'])
  })

  it('applies disabled class when disabled prop is true', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B'],
        disabled: true
      }
    })

    expect(wrapper.classes()).toContain('is-disabled')
  })

  it('applies has-left-icon class when hasLeftIcon prop is true', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B'],
        hasLeftIcon: true
      }
    })

    expect(wrapper.classes()).toContain('has-left-icon')
  })

  it('generates correct option values for primitive options', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['First', 'Second', 'Third']
      }
    })

    expect(wrapper.vm.getOptionValue('First', 0)).toBe('First')
    expect(wrapper.vm.getOptionValue('Second', 1)).toBe('Second')
  })

  it('generates correct option values for object options', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { label: 'First', value: '1' },
          { label: 'Second', value: '2' }
        ]
      }
    })

    expect(wrapper.vm.getOptionValue({ label: 'First', value: '1' }, 0)).toBe('1')
    expect(wrapper.vm.getOptionValue({ label: 'Second', value: '2' }, 1)).toBe('2')
  })

  it('uses optionValue prop to extract values from objects', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { name: 'Item 1', id: 'item1' },
          { name: 'Item 2', id: 'item2' }
        ],
        optionValue: 'id'
      }
    })

    expect(wrapper.vm.getOptionValue({ name: 'Item 1', id: 'item1' }, 0)).toBe('item1')
  })

  it('uses optionLabel prop to extract labels from objects', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { name: 'Item 1', id: 'item1' },
          { name: 'Item 2', id: 'item2' }
        ],
        optionLabel: 'name'
      }
    })

    expect(wrapper.vm.getOptionLabel({ name: 'Item 1', id: 'item1' })).toBe('Item 1')
  })

  it('falls back to index when optionValue is not found', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { name: 'Item 1' },
          { name: 'Item 2' }
        ]
      }
    })

    expect(wrapper.vm.getOptionValue({ name: 'Item 1' }, 0)).toBe(0)
    expect(wrapper.vm.getOptionValue({ name: 'Item 2' }, 1)).toBe(1)
  })

  it('falls back to value or option itself when optionLabel is not found', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { name: 'Item 1', value: 'val1' },
          'plain string'
        ]
      }
    })

    expect(wrapper.vm.getOptionLabel({ name: 'Item 1', value: 'val1' })).toBe('val1')
    expect(wrapper.vm.getOptionLabel('plain string')).toBe('plain string')
  })

  it('computes selectedLabel correctly', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { label: 'Option A', value: 'a' },
          { label: 'Option B', value: 'b' }
        ],
        modelValue: 'b'
      }
    })

    expect(wrapper.vm.selectedLabel).toBe('Option B')
  })

  it('returns null for selectedLabel when no value matches', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: [
          { label: 'Option A', value: 'a' },
          { label: 'Option B', value: 'b' }
        ],
        modelValue: 'c' // doesn't exist
      }
    })

    expect(wrapper.vm.selectedLabel).toBe(null)
  })

  it('returns null for selectedLabel when modelValue is null', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C'],
        modelValue: null
      }
    })

    expect(wrapper.vm.selectedLabel).toBe(null)
  })

  it('toggles dropdown open state', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B']
      }
    })

    expect(wrapper.vm.isOpen).toBe(false)

    wrapper.vm.toggleDropdown()
    expect(wrapper.vm.isOpen).toBe(true)

    wrapper.vm.toggleDropdown()
    expect(wrapper.vm.isOpen).toBe(false)
  })

  it('does not toggle when disabled', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B'],
        disabled: true
      }
    })

    wrapper.vm.toggleDropdown()
    expect(wrapper.vm.isOpen).toBe(false)
  })

  it('closes dropdown when selecting option', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B']
      }
    })

    wrapper.vm.isOpen = true
    wrapper.vm.selectOption('A', 0)

    expect(wrapper.vm.isOpen).toBe(false)
  })

  it('renders trigger button with correct classes', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B']
      }
    })

    const trigger = wrapper.find('.dropdown-trigger')
    expect(trigger.exists()).toBe(true)
    expect(trigger.classes()).toContain('dropdown-trigger')
  })

  it('renders arrow icon', () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B']
      }
    })

    const arrow = wrapper.find('.dropdown-arrow')
    expect(arrow.exists()).toBe(true)
  })

  it('renders dropdown menu when open', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C']
      }
    })

    wrapper.vm.isOpen = true
    await wrapper.vm.$nextTick()

    const menu = wrapper.find('.dropdown-menu')
    expect(menu.exists()).toBe(true)

    const options = wrapper.findAll('.dropdown-option')
    expect(options.length).toBe(3)
  })

  it('applies selected class to selected option', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C'],
        modelValue: 'B'
      }
    })

    wrapper.vm.isOpen = true
    await wrapper.vm.$nextTick()

    const options = wrapper.findAll('.dropdown-option')
    expect(options[1].classes()).toContain('is-selected')
  })

  it('shows checkmark for selected option', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['A', 'B', 'C'],
        modelValue: 'B'
      }
    })

    wrapper.vm.isOpen = true
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('✓')
  })

  it('renders option text correctly', async () => {
    const wrapper = mountCustomDropdown({
      props: {
        options: ['First Option', 'Second Option']
      }
    })

    wrapper.vm.isOpen = true
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('First Option')
    expect(wrapper.text()).toContain('Second Option')
  })
})

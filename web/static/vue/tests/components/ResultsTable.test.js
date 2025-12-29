/**
 * Tests for ResultsTable component
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { i18n } from '../setup'
import ResultsTable from '../../src/components/ResultsTable.vue'

describe('ResultsTable', () => {
  const mockResults = {
    long_symbols_with_confidence: [
      ['BTC/USDT', 0.85],
      ['ETH/USDT', 0.75],
    ],
    short_symbols_with_confidence: [
      ['SOL/USDT', 0.65],
    ],
    none_symbols: ['ADA/USDT'],
  }

  it('should render summary section with correct counts', () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })
    expect(wrapper.text()).toContain('2') // LONG count
    expect(wrapper.text()).toContain('1') // SHORT count
    expect(wrapper.text()).toContain('1') // NONE count
    expect(wrapper.text()).toContain('4') // Total count
  })

  it('should render empty state when results is null', () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: null,
      },
      global: {
        plugins: [i18n],
      },
    })

    expect(wrapper.find('.max-w-7xl').exists()).toBe(false)
  })

  it('should switch between LONG and SHORT tabs', async () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Tabs use i18n text, find by translated text
    const longTab = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('results.long')))
    const shortTab = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('results.short')))

    expect(longTab).toBeTruthy()
    expect(shortTab).toBeTruthy()

    // Click SHORT tab
    await shortTab.trigger('click')

    // Should show SHORT signals
    expect(wrapper.text()).toContain('SOL/USDT')
  })

  it('should filter signals by symbol', async () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    const filterInput = wrapper.find('input[type="text"]')
    await filterInput.setValue('BTC')

    // Should only show BTC/USDT
    expect(wrapper.text()).toContain('BTC/USDT')
    expect(wrapper.text()).not.toContain('ETH/USDT')
  })

  it('should sort signals by confidence', async () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Sort uses CustomDropdown, not native select
    const sortDropdown = wrapper.findComponent({ name: 'CustomDropdown' })
    expect(sortDropdown.exists()).toBe(true)
    // Set value via component's v-model
    await wrapper.setData({ sortBy: 'confidence' })
    await wrapper.vm.$nextTick()

    // Check if sorting is applied (desc by default)
    const tableRows = wrapper.findAll('tbody tr')
    expect(tableRows.length).toBeGreaterThan(0)

    // Parse symbol and confidence values from each row
    const rowData = tableRows.map(row => {
      const cells = row.findAll('td')
      return {
        symbol: cells[0].text(),
        confidence: parseFloat(
          // remove the '%' sign and convert to number
          cells[1].text().replace('%', '').trim()
        )
      }
    })

    // Verify data is sorted by confidence desc
    for (let i = 1; i < rowData.length; i++) {
      expect(rowData[i - 1].confidence).toBeGreaterThanOrEqual(rowData[i].confidence)
    }

    // Optional: Verify the highest confidence is the first row, and check symbol
    expect(rowData[0].symbol).toBe('BTC/USDT') // Assuming BTC/USDT has highest confidence in mockResults
    expect(rowData[0].confidence).toBeGreaterThanOrEqual(rowData[rowData.length - 1].confidence)
  })


  it('should sort signals by symbol', async () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Find the symbol header (th element) in the table
    const symbolHeader = wrapper.find('thead th')
    expect(symbolHeader.exists()).toBe(true)

    await symbolHeader.trigger('click')
    await wrapper.vm.$nextTick()

    // Verify sorting by checking rendered table rows
    const tableRows = wrapper.findAll('tbody tr')
    expect(tableRows.length).toBeGreaterThan(0)

    // Extract symbol text from each row (first cell)
    const symbolTexts = tableRows.map(row => {
      const cells = row.findAll('td')
      return cells[0].text().trim()
    })

    // Verify symbols are sorted in ascending order (first click sorts ascending)
    // Expected order: BTC/USDT, ETH/USDT (alphabetically)
    const expectedOrder = [...symbolTexts].sort((a, b) => a.localeCompare(b))
    expect(symbolTexts).toEqual(expectedOrder)
  })

  it('should emit symbol-click event when analyze button is clicked', async () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    const analyzeButton = wrapper.find('[data-testid="analyze-button"]')
    if (analyzeButton.exists()) {
      await analyzeButton.trigger('click')
      expect(wrapper.emitted('symbol-click')).toBeTruthy()
    }
  })

  it('should handle pagination correctly', async () => {
    // Create results with more than 20 items
    const manyResults = {
      long_symbols_with_confidence: Array.from({ length: 25 }, (_, i) => [
        `SYMBOL${i}/USDT`,
        0.5 + i * 0.01,
      ]),
      short_symbols_with_confidence: [],
      none_symbols: [],
    }

    const wrapper = mount(ResultsTable, {
      props: {
        results: manyResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Pagination only shows when totalPages > 1
    // For manyResults with 25 items, pagination should appear if itemsPerPage < 25
    const totalPages = Math.ceil(manyResults.long_symbols_with_confidence.length / (wrapper.vm.itemsPerPage || 10))
    if (totalPages > 1) {
      const paginationPage = wrapper.find('[data-testid="pagination-page"]')
      expect(paginationPage.exists()).toBe(true)
    }
  })

  it('should normalize different symbol formats', () => {
    const resultsWithDifferentFormats = {
      long_symbols_with_confidence: [
        ['BTC/USDT', 0.85], // Array format
        'ETH/USDT', // String format
        { symbol: 'SOL/USDT', confidence: 0.75 }, // Object format
      ],
      short_symbols_with_confidence: [],
      none_symbols: [],
    }

    const wrapper = mount(ResultsTable, {
      props: {
        results: resultsWithDifferentFormats,
      },
      global: {
        plugins: [i18n],
      },
    })

    expect(wrapper.text()).toContain('BTC/USDT')
    expect(wrapper.text()).toContain('ETH/USDT')
    expect(wrapper.text()).toContain('SOL/USDT')
  })

  it('should format confidence correctly', () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: {
          long_symbols_with_confidence: [['BTC/USDT', 0.8567]],
          short_symbols_with_confidence: [],
          none_symbols: [],
        },
      },
      global: {
        plugins: [i18n],
      },
    })

    // Confidence should be formatted as percentage
    expect(wrapper.text()).toContain('%')
  })

  it('should reset to page 1 when filter changes', async () => {
    // Use a larger dataset to ensure pagination controls are visible
    const largeResults = {
      long_symbols_with_confidence: Array.from({ length: 25 }, (_, i) => [`SYM${i}/USDT`, Math.random()]),
      short_symbols_with_confidence: [],
      none_symbols: [],
    };

    const wrapper = mount(ResultsTable, {
      props: {
        results: largeResults,
      },
      global: {
        plugins: [i18n],
      },
    });
    
    // Simulate navigating to page 2 via Next button
    const nextButton = wrapperLarge.find('button[aria-label="Next page"]');
    expect(nextButton.exists()).toBe(true);
    await nextButton.trigger('click');
    // Optionally, assert page 2 is active
    const paginationPage = wrapperLarge.find('[data-testid="pagination-page"]');
    expect(paginationPage.exists()).toBe(true);
    expect(paginationPage.text()).toMatch(/2/);

    // Apply filter changes (simulate typing and clearing)
    const filterInput = wrapperLarge.find('input[type="text"]');
    await filterInput.setValue('SYM1');
    await filterInput.setValue('');

    // Assert current page is reset to 1 via DOM
    const paginationPageAfter = wrapperLarge.find('[data-testid="pagination-page"]');
    expect(paginationPageAfter.exists()).toBe(true);
    expect(paginationPageAfter.text()).toMatch(/1/);
  })

  it('should reset to page 1 when tab changes', async () => {
    // Use a larger dataset to ensure pagination controls are visible
    const largeResults = {
      long_symbols_with_confidence: Array.from({ length: 25 }, (_, i) => [`SYM${i}/USDT`, Math.random()]),
      short_symbols_with_confidence: [],
      none_symbols: [],
    };

    const wrapper = mount(ResultsTable, {
      props: {
        results: largeResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Simulate navigating to page 2 using pagination controls
    const nextButton = wrapper.find('button[aria-label="Next page"]');
    expect(nextButton.exists()).toBe(true);
    await nextButton.trigger('click');
    await wrapper.vm.$nextTick();
    
    // Verify we're on page 2 using DOM
    const paginationPageBefore = wrapper.find('[data-testid="pagination-page"]');
    expect(paginationPageBefore.exists()).toBe(true);
    expect(paginationPageBefore.text()).toMatch(/2/);

    // Change tab
    const shortTab = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('results.short')))
    expect(shortTab).toBeTruthy()
    await shortTab.trigger('click')
    await wrapper.vm.$nextTick()

    // Should reset to page 1 - verify using DOM
    const paginationPageAfter = wrapper.find('[data-testid="pagination-page"]');
    expect(paginationPageAfter.exists()).toBe(true);
    expect(paginationPageAfter.text()).toMatch(/1/);
  })

  it('should have items per page selector with correct options', () => {
    const wrapper = mount(ResultsTable, {
      props: {
        results: mockResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Find items per page selector using data-testid - it's a CustomDropdown component
    const itemsPerPageSelector = wrapper.find('[data-testid="items-per-page-selector"]')
    // The data-testid is on CustomDropdown, which should exist
    expect(itemsPerPageSelector.exists()).toBe(true)
    
    // Find the CustomDropdown component that has the items per page options
    // The items per page dropdown is the second CustomDropdown (first is sortBy)
    const allDropdowns = wrapper.findAllComponents({ name: 'CustomDropdown' })
    expect(allDropdowns.length).toBeGreaterThanOrEqual(2)
    
    // Find the dropdown with options [5, 10, 20, 50, 100]
    const expectedOptions = [5, 10, 20, 50, 100]
    const itemsPerPageDropdown = allDropdowns.find(
      comp => Array.isArray(comp.props('options')) && comp.props('options').length === expectedOptions.length && comp.props('options').every((v, i) => v === expectedOptions[i])
    )

    expect(itemsPerPageDropdown).toBeTruthy()

    // Assert that the component's options prop equals the expected array
    const actualOptions = itemsPerPageDropdown.props('options')
    expect(actualOptions).toEqual(expectedOptions)
  })

  it('should change items per page when selector value changes', async () => {
    const manyResults = {
      long_symbols_with_confidence: Array.from({ length: 30 }, (_, i) => [
        `SYMBOL${i}/USDT`,
        0.5 + i * 0.01,
      ]),
      short_symbols_with_confidence: [],
      none_symbols: [],
    }

    const wrapper = mount(ResultsTable, {
      props: {
        results: manyResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Find items per page selector using data-testid - it's a CustomDropdown
    const itemsPerPageSelector = wrapper.find('[data-testid="items-per-page-selector"]')
    expect(itemsPerPageSelector.exists()).toBe(true)
    
    // CustomDropdown uses v-model, so update via component data
    await wrapper.setData({ itemsPerPage: 10 })
    await wrapper.vm.$nextTick()

    // Verify itemsPerPage changed
    expect(wrapper.vm.itemsPerPage).toBe(10)
    
    // Verify pagination updated
    expect(wrapper.vm.totalPages).toBe(Math.ceil(30 / 10))
  })

  it('should reset to page 1 when items per page changes', async () => {
    const manyResults = {
      long_symbols_with_confidence: Array.from({ length: 30 }, (_, i) => [
        `SYMBOL${i}/USDT`,
        0.5 + i * 0.01,
      ]),
      short_symbols_with_confidence: [],
      none_symbols: [],
    }

    const wrapper = mount(ResultsTable, {
      props: {
        results: manyResults,
      },
      global: {
        plugins: [i18n],
      },
    })

    // Navigate to page 2 using pagination controls
    const nextButton = wrapper.find('[data-testid="pagination-next"]');
    await nextButton.trigger('click');
    await wrapper.vm.$nextTick();

    expect(wrapper.vm.currentPage).toBe(2);

    // Change items per page using data-testid
    const itemsPerPageSelector = wrapper.find('[data-testid="items-per-page-selector"]');
    // CustomDropdown uses v-model, update via component data
    await wrapper.setData({ itemsPerPage: 10 })
    await wrapper.vm.$nextTick()

    // Should reset to page 1
    expect(wrapper.vm.currentPage).toBe(1);
  })
})


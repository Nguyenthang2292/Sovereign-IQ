/**
 * Tests for useNumberInput composable
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref, computed, nextTick } from 'vue'
import { useNumberInput } from '../../src/composables/useNumberInput'

describe('useNumberInput', () => {
  describe('optional field (defaultValue = null)', () => {
    it('should initialize with empty string when form value is null', () => {
      const formValue = ref(null)
      const { displayValue } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      expect(displayValue.value).toBe('')
    })

    it('should initialize with form value when provided', () => {
      const formValue = ref(100)
      const { displayValue } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      expect(displayValue.value).toBe('100')
    })

    it('should allow empty input and set form value to null', () => {
      const formValue = ref(null)
      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      displayValue.value = ''
      handleInput()

      expect(formValue.value).toBeNull()
    })

    it('should update form value when valid number is entered', () => {
      const formValue = ref(null)
      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
        valueValidator: (num) => num > 0,
      })

      displayValue.value = '50'
      handleInput()

      expect(formValue.value).toBe(50)
    })

    it('should clear on blur when empty', () => {
      const formValue = ref(100)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      displayValue.value = ''
      handleBlur()

      expect(formValue.value).toBeNull()
      expect(displayValue.value).toBe('')
    })

    it('should normalize display value on blur when valid', () => {
      const formValue = ref(null)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      displayValue.value = '100'
      handleBlur()

      expect(formValue.value).toBe(100)
      expect(displayValue.value).toBe('100')
    })

    it('should clear invalid input on blur', () => {
      const formValue = ref(100)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
        valueValidator: (num) => num > 0,
      })

      displayValue.value = '-5'
      handleBlur()

      expect(formValue.value).toBeNull()
      expect(displayValue.value).toBe('')
    })
  })

  describe('required field with default value', () => {
    it('should initialize with default value when form value is null', () => {
      const formValue = ref(null)
      const { displayValue } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 500,
      })

      expect(displayValue.value).toBe('500')
    })

    it('should set default value on blur when empty', () => {
      const formValue = ref(null)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 500,
      })

      displayValue.value = ''
      handleBlur()

      expect(formValue.value).toBe(500)
      expect(displayValue.value).toBe('500')
    })

    it('should reset to default on blur when invalid', () => {
      const formValue = ref(1000)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 500,
        valueValidator: (num) => num > 0,
      })

      displayValue.value = '-10'
      handleBlur()

      expect(formValue.value).toBe(500)
      expect(displayValue.value).toBe('500')
    })

    it('should normalize valid input on blur', () => {
      const formValue = ref(500)
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 500,
      })

      displayValue.value = '1000'
      handleBlur()

      expect(formValue.value).toBe(1000)
      expect(displayValue.value).toBe('1000')
    })
  })

  describe('value validator', () => {
    it('should accept values >= 0 for cooldown', () => {
      const formValue = ref(null)
      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 2.5,
        valueValidator: (num) => num >= 0,
      })

      displayValue.value = '0'
      handleInput()

      expect(formValue.value).toBe(0)
    })

    it('should reject negative values when validator requires > 0', () => {
      const formValue = ref(100)
      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
        valueValidator: (num) => num > 0,
      })

      displayValue.value = '-5'
      handleInput()

      expect(formValue.value).toBeNull()
    })

    it('should handle floating point values', () => {
      const formValue = ref(null)
      const { displayValue, handleInput, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: 2.5,
        valueValidator: (num) => num >= 0,
      })

      displayValue.value = '3.14'
      handleInput()

      expect(formValue.value).toBe(3.14)

      // Test blur normalizes floating point display
      displayValue.value = '5.789'
      handleBlur()

      expect(formValue.value).toBe(5.789)
      expect(displayValue.value).toBe('5.789')
    })
  })

  describe('validator callback', () => {
    it('should call validator after input', () => {
      const formValue = ref(null)
      const validator = vi.fn()
      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
        validator,
      })

      displayValue.value = '100'
      handleInput()

      expect(validator).toHaveBeenCalled()
    })

    it('should call validator after blur', () => {
      const formValue = ref(null)
      const validator = vi.fn()
      const { displayValue, handleBlur } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
        validator,
      })

      displayValue.value = '100'
      handleBlur()

      expect(validator).toHaveBeenCalled()
    })
  })

  describe('sync with external form value changes', () => {
    it('should update display value when form value changes externally', async () => {
      const formValue = ref(100)
      const { displayValue } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      expect(displayValue.value).toBe('100')

      formValue.value = 200
      await nextTick()

      expect(displayValue.value).toBe('200')
    })

    it('should clear display value when form value becomes null (optional field)', async () => {
      const formValue = ref(100)
      const { displayValue } = useNumberInput({
        formValueRef: formValue,
        defaultValue: null,
      })

      formValue.value = null
      await nextTick()

      expect(displayValue.value).toBe('')
    })
  })

  describe('computed ref support', () => {
    it('should work with computed ref', () => {
      const parentForm = ref({ limit: 500 })
      const formValueRef = computed({
        get: () => parentForm.value.limit,
        set: (val) => { parentForm.value.limit = val }
      })

      const { displayValue, handleInput } = useNumberInput({
        formValueRef: formValueRef,
        defaultValue: 500,
      })

      expect(displayValue.value).toBe('500')

      displayValue.value = '1000'
      handleInput()

      expect(parentForm.value.limit).toBe(1000)
    })
  })
})


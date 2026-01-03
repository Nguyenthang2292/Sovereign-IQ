/**
 * Composable for handling number input fields with display string and form value separation.
 * This pattern avoids Vue's v-model.number coercion issues with empty inputs.
 * 
 * @param {Object} options - Configuration options
 * @param {Ref} options.formValueRef - Reference to the form value (e.g., form.value.maxSymbols)
 * @param {number|null} options.defaultValue - Default value for required fields, or null for optional
 * @param {Function} options.validator - Optional validation function to call after updates
 * @param {Function} options.valueValidator - Optional function to validate the numeric value (e.g., num > 0, num >= 0)
 * @returns {Object} Object containing display ref and input/blur handlers
 */
import { ref, watch, isRef } from 'vue'

export function useNumberInput({
  formValueRef,
  defaultValue = null,
  validator = null,
  valueValidator = (num) => num > 0, // Default: positive numbers only
}) {
  // Helper to get current form value
  const getFormValue = () => {
    return formValueRef.value
  }
  
  // Helper to set form value
  const setFormValue = (value) => {
    if (isRef(formValueRef)) {
      formValueRef.value = value
    } else if (formValueRef && typeof formValueRef === 'object' && 'value' in formValueRef) {
      // Computed ref with setter
      formValueRef.value = value
    } else {
      console.error('useNumberInput: formValueRef must be a writable Vue ref. Computed refs must use { get, set } syntax.')
      throw new Error('Invalid formValueRef provided to useNumberInput')
    }
  }

  // Initialize display value from form value
  const initialValue = getFormValue()
  const displayValue = ref(
    initialValue !== null && initialValue !== undefined
      ? initialValue.toString()
      : (defaultValue !== null ? defaultValue.toString() : '')
  )

  // Sync display value when form value changes externally
  watch(() => getFormValue(), (newValue) => {
    if (newValue !== null && newValue !== undefined) {
      displayValue.value = newValue.toString()
    } else if (defaultValue === null) {
      // Optional field: allow empty display
      displayValue.value = ''
    }
  })

  /**
   * Handle input event - update form value if valid, allow free typing
   */
  function handleInput() {
    const value = displayValue.value
    if (value === '' || value === null || value === undefined) {
      setFormValue(defaultValue === null ? null : defaultValue)
    } else {
      const num = Number(value)
      if (!isNaN(num) && valueValidator(num)) {
        setFormValue(num)
      } else {
        setFormValue(defaultValue === null ? null : defaultValue)
      }
    }
    if (validator) {
      validator()
    }
  }

  /**
   * Handle blur event - normalize display value and set defaults if needed
   */
  function handleBlur() {
    const value = displayValue.value
    if (value === '' || value === null || value === undefined) {
      // Empty input
      if (defaultValue === null) {
        // Optional field: keep empty
        setFormValue(null)
        displayValue.value = ''
      } else {
        // Required field: set default
        setFormValue(defaultValue)
        displayValue.value = defaultValue.toString()
      }
    } else {
      const num = Number(value)
      if (!isNaN(num) && valueValidator(num)) {
        // Valid number
        setFormValue(num)
        displayValue.value = num.toString()
      } else {
        // Invalid number
        if (defaultValue === null) {
          // Optional field: clear
          setFormValue(null)
          displayValue.value = ''
        } else {
          // Required field: reset to default
          setFormValue(defaultValue)
          displayValue.value = defaultValue.toString()
        }
      }
    }
    if (validator) {
      validator()
    }
  }

  return {
    displayValue,
    handleInput,
    handleBlur,
  }
}


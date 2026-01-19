/**
 * Utility to parse ANSI escape codes and categorize log levels
 */

export interface AnsiPart {
    text: string
    color: string
}

/**
 * Parse ANSI escape codes and extract text with color information
 * @param {string} text - Text with ANSI escape codes
 * @returns {AnsiPart[]} Array of objects with text and color
 */
export function parseAnsiCodes(text: string): AnsiPart[] {
    if (!text) return []

    // ANSI color codes mapping (handle both full escape sequences and partial codes)
    const ansiColors: Record<string, string> = {
        '[0m': 'reset',
        '[22m': 'reset',
        '[30m': 'black',
        '[31m': 'red',
        '[32m': 'green',
        '[33m': 'yellow',
        '[34m': 'blue',
        '[35m': 'magenta',
        '[36m': 'cyan',
        '[37m': 'white',
        '[90m': 'gray',
        '[91m': 'bright-red',
        '[92m': 'bright-green',
        '[93m': 'bright-yellow',
        '[94m': 'bright-blue',
        '[95m': 'bright-magenta',
        '[96m': 'bright-cyan',
        '[97m': 'bright-white',
    }

    const parts: AnsiPart[] = []
    let currentText = ''
    let currentColor: string | null = null
    let i = 0

    while (i < text.length) {
        // Check for full ANSI escape sequence: \x1b[XXm or \u001b[XXm
        if (text[i] === '\x1b' || text[i] === '\u001b') {
            const codeMatch = text.substring(i).match(/^[\x1b\u001b]\[(\d+)?m/)
            if (codeMatch) {
                // Save current text if any
                if (currentText) {
                    parts.push({
                        text: currentText,
                        color: currentColor || 'default',
                    })
                    currentText = ''
                }

                const code = `[${codeMatch[1] || '0'}m`
                currentColor = ansiColors[code] || null
                i += codeMatch[0].length
                continue
            }
        }

        currentText += text[i]
        i++
    }

    // Add remaining text
    if (currentText) {
        parts.push({
            text: currentText,
            color: currentColor || 'default',
        })
    }

    return parts.length > 0 ? parts : [{ text, color: 'default' }]
}

/**
 * Detect log level from text content
 * @param {string} text - Log text
 * @returns {string} Log level: 'error', 'warning', 'info', 'success', 'debug'
 */
export function detectLogLevel(text: string): string {
    if (!text) return 'info'

    // Error patterns (check first for highest priority)
    if (
        /\b(errors?|failed?|exceptions?|tracebacks?|fatal?|critical?)\b|❌/i.test(text)
    ) {
        return 'error'
    }

    // Warning patterns
    if (
        /\b(warnings?|warns?|caution)\b|⚠️|⚠/i.test(text)
    ) {
        return 'warning'
    }

    // Success patterns (extended for additional tokens and past tenses)
    if (
        /\b(success|successful|successfully|successes?|create[ds]?|complete[ds]?|done|finish(?:ed)?|save[ds]?|processed?|passed?|succeeded?)\b|✅|✓|🎉|🟢|✔️|✔/i.test(text)
    ) {
        return 'success'
    }

    // Debug patterns
    const lowerText = text.toLowerCase()
    if (
        /\bdebugs?\b/i.test(text) ||
        (lowerText.startsWith('[') && lowerText.includes(']') && text.length < 50)
    ) {
        return 'debug'
    }

    // Check for common info patterns
    if (
        /\b(analyzing|sending|processing|loading|fetching)\b/i.test(text)
    ) {
        return 'info'
    }

    // Default to info
    return 'info'
}

/**
 * Get CSS class for log level
 * @param {string} level - Log level
 * @returns {string} CSS class name
 */
export function getLogLevelClass(level: string): string {
    const classes: Record<string, string> = {
        error: 'text-red-400',
        warning: 'text-yellow-400',
        info: 'text-blue-400',
        success: 'text-green-400',
        debug: 'text-gray-400',
        default: 'text-gray-300',
    }
    return classes[level] || classes.default
}

/**
 * Get CSS class for ANSI color
 * @param {string} color - ANSI color name
 * @returns {string} CSS class name
 */
export function getAnsiColorClass(color: string): string {
    const classes: Record<string, string> = {
        red: 'text-red-400',
        'bright-red': 'text-red-300',
        green: 'text-green-400',
        'bright-green': 'text-green-300',
        yellow: 'text-yellow-400',
        'bright-yellow': 'text-yellow-300',
        blue: 'text-blue-400',
        'bright-blue': 'text-blue-300',
        cyan: 'text-cyan-400',
        'bright-cyan': 'text-cyan-300',
        magenta: 'text-purple-400',
        'bright-magenta': 'text-purple-300',
        white: 'text-white',
        'bright-white': 'text-gray-100',
        gray: 'text-gray-400',
        black: 'text-gray-600',
        reset: 'text-gray-300',
        default: 'text-gray-300',
    }
    return classes[color] || classes.default
}

/**
 * Clean ANSI codes from text
 * @param {string} text - Text with ANSI codes
 * @returns {string} Clean text
 */
export function cleanAnsiCodes(text: string): string {
    if (!text) return ''
    // Remove only valid ANSI escape sequences, which always start with ESC or \u001b
    return text.replace(/[\x1b\u001b]\[[0-9;]*m/g, '')
}

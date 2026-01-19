import { SeriesMarker } from 'lightweight-charts'

interface SignalPoint {
    x: number
    y: number
}

export function useChartHelpers() {
    const getMAColor = (maType: string): string => {
        const colors: Record<string, string> = {
            EMA: '#00E396',
            HMA: '#FEB019',
            WMA: '#775DD0',
            DEMA: '#008FFB',
            LSMA: '#FF4560',
            KAMA: '#775DD0'
        }
        return colors[maType] || '#888'
    }

    const getSignalColor = (signalType: string): string => {
        if (signalType === 'Average_Signal') return '#FFFFFF'
        const colors: Record<string, string> = {
            EMA_Signal: '#00E396',
            HMA_Signal: '#FEB019',
            WMA_Signal: '#775DD0',
            DEMA_Signal: '#008FFB',
            LSMA_Signal: '#FF4560',
            KAMA_Signal: '#775DD0'
        }
        return colors[signalType] || '#888'
    }

    // CORRECT IMPLEMENTATION based on original JS
    const getSignalMarkersOriginal = (signalData: SignalPoint[], _color?: string): any[] => {
        if (!signalData) return []

        const markers: any[] = []

        signalData.forEach(point => {
            if (point.y >= 0.5) {
                markers.push({
                    x: point.x,
                    y: point.y,
                    marker: {
                        size: 10,
                        shape: 'triangle-up',
                        fillColor: '#00E396',
                        strokeColor: '#00E396',
                        radius: 2
                    }
                })
            } else if (point.y <= -0.5) {
                markers.push({
                    x: point.x,
                    y: point.y,
                    marker: {
                        size: 10,
                        shape: 'triangle-down',
                        fillColor: '#FF4560',
                        strokeColor: '#FF4560',
                        radius: 2
                    }
                })
            }
        })

        return markers
    }


    const formatTimestamp = (timestamp: number | string | Date): string => {
        const date = new Date(timestamp)
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    const formatPrice = (price: number | null | undefined): string => {
        if (price === null || price === undefined) return 'N/A'

        if (price >= 1000) {
            return price.toFixed(2)
        } else if (price >= 1) {
            return price.toFixed(4)
        } else {
            return price.toFixed(8)
        }
    }

    return {
        getMAColor,
        getSignalColor,
        getSignalMarkers: getSignalMarkersOriginal,
        formatTimestamp,
        formatPrice
    }
}

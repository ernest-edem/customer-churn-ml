const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

/**
 * Send a customer assessment to the FastAPI prediction endpoint.
 *
 * @param {Object} assessmentData
 * @returns {Promise<Object>} Prediction response from the API
 */
export async function predictChurn(assessmentData) {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(assessmentData),
        })

        if (!response.ok) {
            let errorMessage = `Prediction request failed with status ${response.status}.`

            try {
                const errorData = await response.json()

                if (errorData?.detail) {
                    errorMessage =
                        typeof errorData.detail === 'string'
                            ? errorData.detail
                            : JSON.stringify(errorData.detail)
                }
            } catch {
                // Keep the default HTTP error message when the response is not JSON.
            }

            throw new Error(errorMessage)
        }

        return await response.json()
    } catch (error) {
        if (error instanceof Error && error.message !== 'Failed to fetch') {
            throw error
        }

        throw new Error(
            'Unable to connect to the prediction service. Please make sure the FastAPI backend is running and try again.',
            { cause: error },
        )
    }
}
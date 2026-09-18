import { AlertCircle, ArrowLeft, BrainCircuit, LoaderCircle } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import CustomerAssessmentForm from '../components/CustomerAssessmentForm'
import { predictChurn } from '../services/predictionService'

function AssessmentPage() {
    const navigate = useNavigate()
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [error, setError] = useState('')

    const handleAssessmentSubmit = async (assessmentData) => {
        setIsSubmitting(true)
        setError('')

        try {
            const prediction = await predictChurn(assessmentData)

            navigate('/results', {
                state: {
                    assessment: assessmentData,
                    prediction,
                },
            })
        } catch (requestError) {
            setError(
                requestError instanceof Error
                    ? requestError.message
                    : 'Unable to generate a churn prediction. Please try again.',
            )
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <div className="min-h-screen bg-background text-text">
            <header className="border-b border-border bg-surface">
                <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-8">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-white">
                            <BrainCircuit size={22} strokeWidth={2} />
                        </div>

                        <div>
                            <h1 className="text-lg font-semibold tracking-tight">
                                Customer Churn ML
                            </h1>

                            <p className="text-sm text-text-muted">
                                Customer assessment
                            </p>
                        </div>
                    </div>

                    <button
                        type="button"
                        onClick={() => navigate('/')}
                        disabled={isSubmitting}
                        className="hidden items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:cursor-not-allowed disabled:opacity-60 sm:inline-flex"
                    >
                        <ArrowLeft size={16} />
                        Dashboard
                    </button>
                </div>
            </header>

            <main className="mx-auto max-w-5xl px-6 py-8 lg:px-8">
                {error && (
                    <div
                        role="alert"
                        className="mb-6 flex items-start gap-3 rounded-xl border border-danger/20 bg-danger/5 p-4"
                    >
                        <AlertCircle
                            className="mt-0.5 shrink-0 text-danger"
                            size={20}
                        />

                        <div>
                            <h2 className="text-sm font-semibold text-text">
                                Prediction failed
                            </h2>

                            <p className="mt-1 text-sm leading-6 text-text-muted">
                                {error}
                            </p>

                            <p className="mt-2 text-xs text-text-muted">
                                Your assessment is still available. Please try submitting it
                                again.
                            </p>
                        </div>
                    </div>
                )}

                <CustomerAssessmentForm
                    onSubmit={handleAssessmentSubmit}
                    onCancel={() => navigate('/')}
                />
            </main>

            {isSubmitting && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-text/20 px-6 backdrop-blur-sm">
                    <div
                        role="status"
                        aria-live="polite"
                        className="flex w-full max-w-sm flex-col items-center rounded-2xl border border-border bg-surface p-8 text-center shadow-xl"
                    >
                        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary-soft text-primary">
                            <LoaderCircle className="animate-spin" size={24} />
                        </div>

                        <h2 className="mt-5 text-lg font-semibold text-text">
                            Generating prediction
                        </h2>

                        <p className="mt-2 text-sm leading-6 text-text-muted">
                            Sending the customer assessment to the churn prediction model.
                        </p>
                    </div>
                </div>
            )}
        </div>
    )
}

export default AssessmentPage
import {
    BarChart3,
    BrainCircuit,
    CircleCheck,
    Database,
    LayoutDashboard,
    Users,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'

function LandingPage({ onStartAssessment }) {
    const navigate = useNavigate()

    return (
        <div className="min-h-screen bg-background text-text">
            <header className="border-b border-border bg-surface">
                <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-8">
                    <div className="flex items-center gap-3">
                        <img
                            src="/favicon.svg"
                            alt="Customer Churn ML logo"
                            className="h-10 w-10 rounded-xl object-cover"
                        />

                        <div>
                            <h1 className="text-lg font-semibold tracking-tight">
                                Customer Churn ML
                            </h1>
                            <p className="text-sm text-text-muted">
                                Customer retention analysis
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <span className="hidden items-center gap-2 rounded-full border border-success/20 bg-success/5 px-3 py-1.5 text-xs font-semibold text-success sm:inline-flex">
                            <CircleCheck size={14} />
                            API Ready
                        </span>

                        <button
                            type="button"
                            onClick={() => navigate('/model')}
                            className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            <BarChart3 size={16} />
                            Model Information
                        </button>
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
                <div className="mb-8">
                    <p className="text-sm font-semibold text-primary">
                        Customer Analytics
                    </p>

                    <h2 className="mt-1 text-3xl font-bold tracking-tight text-text">
                        Customer Churn Prediction
                    </h2>

                    <p className="mt-2 max-w-2xl text-sm leading-6 text-text-muted">
                        Use the machine learning model to assess customer churn risk and
                        review the prediction results.
                    </p>
                </div>

                <section className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-xl border border-border bg-surface p-5">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-soft text-primary">
                            <Users size={19} />
                        </div>

                        <h3 className="mt-4 font-semibold text-text">
                            Customer Assessment Ready
                        </h3>

                        <p className="mt-1 text-sm leading-6 text-text-muted">
                            Enter customer profile, service, billing, and usage information
                            to generate a churn prediction.
                        </p>
                    </div>

                    <div className="rounded-xl border border-border bg-surface p-5">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-soft text-primary">
                            <BrainCircuit size={19} />
                        </div>

                        <h3 className="mt-4 font-semibold text-text">
                            Prediction Model
                        </h3>

                        <p className="mt-1 text-sm leading-6 text-text-muted">
                            Gradient Boosting classification model trained on customer churn
                            data.
                        </p>
                    </div>

                    <div className="rounded-xl border border-border bg-surface p-5">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-soft text-primary">
                            <BarChart3 size={19} />
                        </div>

                        <h3 className="mt-4 text-2xl font-bold tracking-tight text-text">
                            83.44%
                        </h3>

                        <p className="mt-1 text-sm text-text-muted">
                            Holdout test performance
                        </p>
                    </div>
                </section>

                <section className="mt-8 rounded-2xl border border-border bg-surface p-6 shadow-sm">
                    <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
                        <div>
                            <div className="flex items-center gap-2">
                                <LayoutDashboard size={18} className="text-primary" />

                                <h3 className="font-semibold text-text">
                                    Start a customer assessment
                                </h3>
                            </div>

                            <p className="mt-2 max-w-2xl text-sm leading-6 text-text-muted">
                                Complete the assessment form and send the customer data to the
                                prediction API.
                            </p>
                        </div>

                        <button
                            type="button"
                            onClick={onStartAssessment}
                            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            New Assessment
                        </button>
                    </div>
                </section>

                <section className="mt-8">
                    <div className="mb-4 flex items-center gap-2">
                        <Database size={18} className="text-primary" />

                        <h3 className="font-semibold text-text">
                            System Status
                        </h3>
                    </div>

                    <div className="grid gap-4 md:grid-cols-3">
                        <div className="rounded-xl border border-border bg-surface p-4">
                            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                                Frontend
                            </p>

                            <p className="mt-2 flex items-center gap-2 text-sm font-semibold text-success">
                                <CircleCheck size={16} />
                                Online
                            </p>
                        </div>

                        <div className="rounded-xl border border-border bg-surface p-4">
                            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                                Prediction API
                            </p>

                            <p className="mt-2 flex items-center gap-2 text-sm font-semibold text-success">
                                <CircleCheck size={16} />
                                Ready
                            </p>
                        </div>

                        <div className="rounded-xl border border-border bg-surface p-4">
                            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                                Model
                            </p>

                            <p className="mt-2 flex items-center gap-2 text-sm font-semibold text-success">
                                <CircleCheck size={16} />
                                Loaded
                            </p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    )
}

export default LandingPage
import {
    AlertCircle,
    ArrowLeft,
    BarChart3,
    BrainCircuit,
    CheckCircle2,
    CircleAlert,
    RotateCcw,
} from 'lucide-react'
import { useLocation, useNavigate } from 'react-router-dom'

function getRiskLevel(probability) {
    if (probability >= 0.7) {
        return {
            label: 'High risk',
            description:
                'The model estimates a relatively high probability that this customer may churn.',
            icon: CircleAlert,
            className: 'text-danger',
            badgeClassName: 'border-danger/20 bg-danger/5 text-danger',
        }
    }

    if (probability >= 0.4) {
        return {
            label: 'Moderate risk',
            description:
                'The model estimates a moderate probability that this customer may churn.',
            icon: AlertCircle,
            className: 'text-warning',
            badgeClassName: 'border-warning/20 bg-warning/5 text-warning',
        }
    }

    return {
        label: 'Lower risk',
        description:
            'The model estimates a relatively lower probability that this customer may churn.',
        icon: CheckCircle2,
        className: 'text-success',
        badgeClassName: 'border-success/20 bg-success/5 text-success',
    }
}

function SummarySection({ title, items }) {
    return (
        <div>
            <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                {title}
            </h4>

            <dl className="mt-3 grid gap-x-6 gap-y-3 sm:grid-cols-2">
                {items.map(([label, value]) => (
                    <div
                        key={label}
                        className="flex items-baseline justify-between gap-4 border-b border-border pb-2 last:border-b-0 sm:block"
                    >
                        <dt className="text-sm text-text-muted">{label}</dt>
                        <dd className="text-sm font-medium text-text">{value}</dd>
                    </div>
                ))}
            </dl>
        </div>
    )
}

function ResultsPage() {
    const location = useLocation()
    const navigate = useNavigate()

    const assessment = location.state?.assessment
    const prediction = location.state?.prediction

    if (!assessment || !prediction) {
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
                                    Prediction results
                                </p>
                            </div>
                        </div>

                        <button
                            type="button"
                            onClick={() => navigate('/')}
                            className="hidden items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30 sm:inline-flex"
                        >
                            <ArrowLeft size={16} />
                            Dashboard
                        </button>
                    </div>
                </header>

                <main className="mx-auto flex min-h-[calc(100vh-89px)] max-w-2xl items-center px-6 py-10 lg:px-8">
                    <div className="w-full rounded-2xl border border-border bg-surface p-8 text-center shadow-sm">
                        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary-soft text-primary">
                            <BarChart3 size={24} />
                        </div>

                        <h2 className="mt-5 text-2xl font-bold tracking-tight text-text">
                            No prediction available
                        </h2>

                        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-text-muted">
                            Complete a customer assessment first to generate and view a churn
                            prediction.
                        </p>

                        <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
                            <button
                                type="button"
                                onClick={() => navigate('/assessment')}
                                className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                            >
                                <RotateCcw size={16} />
                                New Assessment
                            </button>

                            <button
                                type="button"
                                onClick={() =>
                                    navigate('/model', {
                                        state: {
                                            assessment,
                                            prediction,
                                        },
                                    })
                                }
                                className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface px-5 py-3 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                            >
                                <BarChart3 size={16} />
                                Model Information
                            </button>
                        </div>
                    </div>
                </main>
            </div>
        )
    }

    const predictionLabel = prediction.prediction
    const churnProbability = prediction.churn_probability

    const probabilityPercentage =
        typeof churnProbability === 'number'
            ? churnProbability * 100
            : null

    const risk =
        typeof churnProbability === 'number'
            ? getRiskLevel(churnProbability)
            : {
                label: 'Unavailable',
                description:
                    'The prediction response did not include a valid churn probability.',
                icon: AlertCircle,
                className: 'text-warning',
                badgeClassName: 'border-warning/20 bg-warning/5 text-warning',
            }

    const RiskIcon = risk.icon

    const profileItems = [
        ['Gender', assessment.gender],
        ['Senior citizen', assessment.SeniorCitizen ? 'Yes' : 'No'],
        ['Partner', assessment.Partner],
        ['Dependents', assessment.Dependents],
        ['Tenure', `${assessment.tenure} months`],
    ]

    const serviceItems = [
        ['Phone service', assessment.PhoneService],
        ['Multiple lines', assessment.MultipleLines],
        ['Internet service', assessment.InternetService],
        ['Online security', assessment.OnlineSecurity],
        ['Online backup', assessment.OnlineBackup],
        ['Device protection', assessment.DeviceProtection],
        ['Tech support', assessment.TechSupport],
    ]

    const billingItems = [
        ['Streaming TV', assessment.StreamingTV],
        ['Streaming movies', assessment.StreamingMovies],
        ['Contract', assessment.Contract],
        ['Paperless billing', assessment.PaperlessBilling],
        ['Payment method', assessment.PaymentMethod],
        ['Monthly charges', `₹${Number(assessment.MonthlyCharges).toFixed(2)}`],
        ['Total charges', `₹${Number(assessment.TotalCharges).toFixed(2)}`],
    ]

    return (
        <div className="min-h-screen bg-background text-text">
            <header className="border-b border-border bg-surface">
                <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-5 lg:px-8 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-white">
                            <BrainCircuit size={22} strokeWidth={2} />
                        </div>

                        <div>
                            <h1 className="text-lg font-semibold tracking-tight">
                                Customer Churn ML
                            </h1>
                            <p className="text-sm text-text-muted">
                                Prediction results
                            </p>
                        </div>
                    </div>

                    <nav
                        aria-label="Results page navigation"
                        className="flex flex-wrap items-center gap-2"
                    >
                        <button
                            type="button"
                            onClick={() => navigate('/')}
                            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Dashboard
                        </button>

                        <button
                            type="button"
                            onClick={() =>
                                navigate('/model', {
                                    state: {
                                        assessment,
                                        prediction,
                                    },
                                })
                            }
                            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Model Information
                        </button>

                        <button
                            type="button"
                            onClick={() => navigate('/assessment')}
                            className="inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            <RotateCcw size={15} />
                            New Assessment
                        </button>
                    </nav>
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
                <div className="mb-8">
                    <p className="text-sm font-semibold text-primary">
                        Prediction complete
                    </p>

                    <h2 className="mt-1 text-3xl font-bold tracking-tight text-text">
                        Customer Churn Result
                    </h2>

                    <p className="mt-2 max-w-3xl text-sm leading-6 text-text-muted">
                        The trained Gradient Boosting model has evaluated the submitted
                        customer assessment.
                    </p>
                </div>

                <section className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
                    <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
                        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                                <p className="text-sm font-semibold text-text-muted">
                                    Prediction
                                </p>

                                <p className="mt-2 text-4xl font-bold tracking-tight text-text">
                                    {predictionLabel === 'Yes'
                                        ? 'Likely to churn'
                                        : 'Likely to stay'}
                                </p>

                                <p className="mt-2 text-sm leading-6 text-text-muted">
                                    Predicted class:{' '}
                                    <span className="font-semibold text-text">
                                        {predictionLabel}
                                    </span>
                                </p>
                            </div>

                            <div
                                className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-full border ${risk.badgeClassName}`}
                            >
                                <RiskIcon size={32} className={risk.className} />
                            </div>
                        </div>

                        <div className="mt-8">
                            <div className="flex items-center justify-between gap-4">
                                <p className="text-sm font-semibold text-text">
                                    Churn probability
                                </p>

                                <p className="text-lg font-bold text-text">
                                    {probabilityPercentage !== null
                                        ? `${probabilityPercentage.toFixed(1)}%`
                                        : 'Unavailable'}
                                </p>
                            </div>

                            {probabilityPercentage !== null && (
                                <div className="mt-3 h-3 overflow-hidden rounded-full bg-surface-muted">
                                    <div
                                        className="h-full rounded-full bg-primary transition-all"
                                        style={{
                                            width: `${Math.min(
                                                Math.max(probabilityPercentage, 0),
                                                100,
                                            )}%`,
                                        }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
                        <p className="text-sm font-semibold text-text-muted">
                            Risk interpretation
                        </p>

                        <div
                            className={`mt-4 inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold ${risk.badgeClassName}`}
                        >
                            <RiskIcon size={16} />
                            {risk.label}
                        </div>

                        <p className="mt-4 text-sm leading-6 text-text-muted">
                            {risk.description}
                        </p>

                        <div className="mt-6 flex items-center gap-3 border-t border-border pt-5">
                            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-soft text-primary">
                                <BrainCircuit size={18} />
                            </div>

                            <div>
                                <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                                    Model
                                </p>
                                <p className="text-sm font-semibold text-text">
                                    Gradient Boosting
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                <section className="mt-8 rounded-2xl border border-border bg-surface p-6 shadow-sm">
                    <div className="mb-6">
                        <h3 className="text-lg font-semibold text-text">
                            Customer Assessment
                        </h3>

                        <p className="mt-1 text-sm text-text-muted">
                            Compact summary of the information submitted for this prediction.
                        </p>
                    </div>

                    <div className="grid gap-8 md:grid-cols-3">
                        <SummarySection title="Customer Profile" items={profileItems} />
                        <SummarySection title="Services" items={serviceItems} />
                        <SummarySection title="Billing & Usage" items={billingItems} />
                    </div>
                </section>

                <section className="mt-8 rounded-xl border border-border bg-surface-muted p-4">
                    <div className="flex items-start gap-3">
                        <AlertCircle
                            size={18}
                            className="mt-0.5 shrink-0 text-text-muted"
                        />

                        <p className="text-sm leading-6 text-text-muted">
                            This prediction is generated by a machine learning model for
                            portfolio and analytical purposes. It should not be treated as a
                            guarantee of future customer behavior.
                        </p>
                    </div>
                </section>

                <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-end">
                    <button
                        type="button"
                        onClick={() => navigate('/model')}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                        <BarChart3 size={16} />
                        View Model Information
                    </button>

                    <button
                        type="button"
                        onClick={() => navigate('/assessment')}
                        className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                        <RotateCcw size={16} />
                        New Assessment
                    </button>
                </div>
            </main>
        </div>
    )
}

export default ResultsPage